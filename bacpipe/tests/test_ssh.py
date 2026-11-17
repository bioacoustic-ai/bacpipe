"""Tests of reading the audio of the dashboard from a server via ssh."""

import sys
from types import ModuleType, SimpleNamespace

import fsspec
import numpy as np
import pytest
import soundfile

from bacpipe.embedding_evaluation.visualization.visualize_spectrograms import (
    SpectrogramPlot,
)

SSH_CONFIGS = {
    "host": "example.cluster.com",
    "port": 22,
    "username": "UserName",
    "key_filename": "path/to/ssh/file",
}


class FakeProxyCommand:
    def __init__(self, command):
        self.command = command


@pytest.fixture
def fake_paramiko(monkeypatch):
    module = ModuleType("paramiko")
    module.ProxyCommand = FakeProxyCommand
    monkeypatch.setitem(sys.modules, "paramiko", module)
    return module


@pytest.fixture
def opened_filesystems(monkeypatch):
    calls = []

    def filesystem(protocol, **kwargs):
        calls.append((protocol, kwargs))
        return SimpleNamespace(protocol=protocol, kwargs=kwargs)

    monkeypatch.setattr(fsspec, "filesystem", filesystem)
    return calls


def connect(ssh_configs):
    plot = object.__new__(SpectrogramPlot)
    plot._initiate_ssh_connection({"ssh_configs": ssh_configs})
    return plot


class TestConnection:
    def test_without_cmd_connects_directly(self, fake_paramiko, opened_filesystems):
        connect(dict(SSH_CONFIGS))
        assert opened_filesystems == [("sftp", SSH_CONFIGS)]

    def test_empty_cmd_connects_directly(self, fake_paramiko, opened_filesystems):
        connect({**SSH_CONFIGS, "cmd": ""})
        assert opened_filesystems == [("sftp", SSH_CONFIGS)]

    def test_cmd_is_used_as_proxy_jump(self, fake_paramiko, opened_filesystems):
        cmd = "ssh -q -o BatchMode=yes -p 22 UserName@login.cluster.com -W node:22"
        plot = connect({**SSH_CONFIGS, "cmd": cmd})

        protocol, kwargs = opened_filesystems[0]
        assert protocol == "sftp"
        assert "cmd" not in kwargs
        assert isinstance(kwargs["sock"], FakeProxyCommand)
        assert kwargs["sock"].command == cmd
        assert {k: v for k, v in kwargs.items() if k != "sock"} == SSH_CONFIGS
        assert plot.ssh_connector.kwargs is kwargs

    def test_the_settings_are_not_changed(self, fake_paramiko, opened_filesystems):
        ssh_configs = {**SSH_CONFIGS, "cmd": "ssh -W node:22 login"}
        connect(ssh_configs)
        assert ssh_configs == {**SSH_CONFIGS, "cmd": "ssh -W node:22 login"}

    def test_missing_paramiko_raises_a_helpful_error(self, monkeypatch, opened_filesystems):
        monkeypatch.setitem(sys.modules, "paramiko", None)
        with pytest.raises(ImportError, match="paramiko"):
            connect(dict(SSH_CONFIGS))
        assert opened_filesystems == []


class TestConnectionIsOnlyOpenedForSsh:
    @staticmethod
    def make_plot(audio_dir, monkeypatch):
        opened = []
        monkeypatch.setattr(
            SpectrogramPlot,
            "_initiate_ssh_connection",
            lambda self, kwargs: opened.append(kwargs["ssh_configs"]),
        )
        metadata = {"sample_rate (Hz)": 16000, "segment_length (samples)": 16000}
        loader = SimpleNamespace(
            get_data=lambda model, label: None,
            embeds={"model_a": {"metadata": metadata}},
        )
        SpectrogramPlot(
            audio_dir,
            loader,
            SimpleNamespace(options=["model_a"]),
            panel_static_text=None,
            paths=None,
            ssh_configs=SSH_CONFIGS,
        )
        return opened

    def test_ssh_audio_dir_opens_the_connection(self, monkeypatch):
        assert self.make_plot("ssh /data/audio", monkeypatch) == [SSH_CONFIGS]

    def test_local_audio_dir_does_not(self, monkeypatch, tmp_path):
        assert self.make_plot(str(tmp_path), monkeypatch) == []
        assert self.make_plot(tmp_path, monkeypatch) == []


class TestRemotePath:
    @staticmethod
    def remote_path(audio_dir, filename):
        plot = object.__new__(SpectrogramPlot)
        plot.audio_dir = audio_dir
        return plot._remote_audio_path(filename)

    @pytest.mark.parametrize(
        "audio_dir, filename, expected",
        [
            ("ssh /data/audio", "rec.wav", "/data/audio/rec.wav"),
            ("ssh /data/audio/", "rec.wav", "/data/audio/rec.wav"),
            ("ssh /data/audio", "site_1/rec.wav", "/data/audio/site_1/rec.wav"),
            ("ssh /data/audio", "site_1\\rec.wav", "/data/audio/site_1/rec.wav"),
            ("ssh /data/ssh audio", "rec.wav", "/data/ssh audio/rec.wav"),
            ("ssh ~/audio", "rec.wav", "~/audio/rec.wav"),
        ],
    )
    def test_is_a_posix_path_on_the_server(self, audio_dir, filename, expected):
        assert self.remote_path(audio_dir, filename) == expected


SAMPLE_RATE_FILE = 16000
SAMPLE_RATE_MODEL = 8000


@pytest.fixture
def audio_folder(tmp_path):
    rng = np.random.default_rng(0)
    signal = rng.uniform(-0.5, 0.5, 2 * SAMPLE_RATE_FILE).astype(np.float32)
    (tmp_path / "site_1").mkdir()
    soundfile.write(tmp_path / "site_1" / "mono.wav", signal, SAMPLE_RATE_FILE)
    stereo = np.stack([signal, -signal * 0.5], axis=1)
    soundfile.write(tmp_path / "site_1" / "stereo.wav", stereo, SAMPLE_RATE_FILE)
    return tmp_path


def make_plot(audio_dir, connector=None, segment_seconds=0.5):
    plot = object.__new__(SpectrogramPlot)
    plot.audio_dir = audio_dir
    plot.sample_rate = SAMPLE_RATE_MODEL
    plot.segment_length = int(segment_seconds * SAMPLE_RATE_MODEL)
    plot.kwargs = {"bool_change_speed": False, "padding": "constant"}
    if connector is not None:
        plot.ssh_connector = connector
    return plot


class RecordingFilesystem:
    """A local filesystem that remembers which paths were opened."""

    def __init__(self):
        self.local = fsspec.filesystem("file")
        self.opened = []

    def open(self, path, mode):
        self.opened.append(path)
        return self.local.open(path, mode)


class TestLoadAudioViaSsh:
    def test_same_audio_as_reading_the_local_file(self, audio_folder):
        connector = RecordingFilesystem()
        remote = make_plot(f"ssh {audio_folder.as_posix()}", connector)
        local = make_plot(audio_folder)

        audio, stem = remote.load_audio(0.25, 0.75, "site_1/mono.wav")
        expected, expected_stem = local.load_audio(0.25, 0.75, "site_1/mono.wav")

        assert connector.opened == [f"{audio_folder.as_posix()}/site_1/mono.wav"]
        assert stem == expected_stem == "mono"
        assert audio.shape == expected.shape == (int(0.5 * SAMPLE_RATE_MODEL),)
        np.testing.assert_allclose(audio, expected, atol=1e-3)

    @pytest.mark.parametrize("start, end", [(0.0, 0.5), (0.3, 1.8), (1.0, 2.0)])
    def test_segments_have_the_requested_length(self, audio_folder, start, end):
        plot = make_plot(
            f"ssh {audio_folder.as_posix()}", RecordingFilesystem(), segment_seconds=end - start
        )
        audio, _ = plot.load_audio(start, end, "site_1/mono.wav")
        assert len(audio) == int(round((end - start) * SAMPLE_RATE_MODEL))

    def test_short_segments_are_padded_to_the_model_length(self, audio_folder):
        plot = make_plot(f"ssh {audio_folder.as_posix()}", RecordingFilesystem(), segment_seconds=1)
        audio, _ = plot.load_audio(1.5, 2.0, "site_1/mono.wav")
        assert len(audio) == SAMPLE_RATE_MODEL

    def test_stereo_files_are_mixed_to_mono(self, audio_folder):
        connector = RecordingFilesystem()
        remote = make_plot(f"ssh {audio_folder.as_posix()}", connector)
        local = make_plot(audio_folder)

        audio, _ = remote.load_audio(0.25, 0.75, "site_1/stereo.wav")
        expected, _ = local.load_audio(0.25, 0.75, "site_1/stereo.wav")

        assert audio.ndim == 1
        np.testing.assert_allclose(audio, expected, atol=1e-3)

    def test_windows_style_file_names_are_read_from_a_posix_path(self, audio_folder):
        connector = RecordingFilesystem()
        plot = make_plot(f"ssh {audio_folder.as_posix()}", connector)
        plot.load_audio(0.0, 0.5, "site_1\\mono.wav")
        assert connector.opened == [f"{audio_folder.as_posix()}/site_1/mono.wav"]

    def test_a_failed_read_raises_the_original_error(self, audio_folder):
        plot = make_plot(f"ssh {audio_folder.as_posix()}", RecordingFilesystem())
        with pytest.raises(FileNotFoundError):
            plot.load_audio(0.0, 0.5, "site_1/missing.wav")

    def test_local_audio_dir_does_not_use_the_connection(self, audio_folder):
        connector = RecordingFilesystem()
        plot = make_plot(audio_folder, connector)
        plot.load_audio(0.0, 0.5, "site_1/mono.wav")
        assert connector.opened == []
