/*
 * bacpipe.exe - the launcher of the bacpipe application on Windows.
 *
 * It does one thing: start the Python that is shipped in the "runtime"
 * folder next to this file,
 *
 *     runtime\pythonw.exe -E -s -X utf8 -m bacpipe_app <arguments>
 *
 * and wait for it, so that the exit code of the application is the exit
 * code of bacpipe.exe.
 *
 *   -E        ignore PYTHONPATH / PYTHONHOME of the user's machine
 *   -s        ignore packages installed in the user's home directory
 *   -X utf8   use UTF-8 for files and output regardless of the system locale
 *
 * pythonw.exe is the Python without a console window. To see a console (for
 * debugging) use bacpipe_debug.cmd, which starts python.exe instead.
 *
 * The launcher needs no installation and no administrator rights: it does
 * not write anything and only reads its own location.
 *
 * It is compiled by app/build.py with the zig compiler (pip package
 * "ziglang"), which works on Windows, Linux and macOS:
 *
 *     python -m ziglang cc launcher.c launcher.rc -o bacpipe.exe
 *            -target x86_64-windows-gnu -municode -Wl,--subsystem,windows -O2 -s
 */

#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <wchar.h>

#define PATH_CHARS 32768

static void show_error(const wchar_t *message, const wchar_t *detail)
{
    wchar_t text[PATH_CHARS + 512];
    _snwprintf(text, sizeof(text) / sizeof(text[0]) - 1, L"%ls\n\n%ls", message, detail);
    text[sizeof(text) / sizeof(text[0]) - 1] = 0;
    MessageBoxW(NULL, text, L"bacpipe", MB_OK | MB_ICONERROR);
}

/* The command line of this process after the program name, i.e. the
 * arguments exactly as they were typed (quoting is kept as it is). */
static const wchar_t *arguments_after_program_name(void)
{
    const wchar_t *p = GetCommandLineW();
    if (*p == L'"') {
        p++;
        while (*p && *p != L'"') p++;
        if (*p == L'"') p++;
    } else {
        while (*p && *p != L' ' && *p != L'\t') p++;
    }
    while (*p == L' ' || *p == L'\t') p++;
    return p;
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR cmdline, int show)
{
    static wchar_t app_dir[PATH_CHARS];
    static wchar_t python[PATH_CHARS];
    static wchar_t command[PATH_CHARS];
    (void)instance; (void)previous; (void)cmdline; (void)show;

    /* 1. the folder this exe is in */
    DWORD length = GetModuleFileNameW(NULL, app_dir, PATH_CHARS);
    if (length == 0 || length >= PATH_CHARS) {
        show_error(L"Could not determine the location of bacpipe.exe.", L"");
        return 1;
    }
    wchar_t *last_separator = wcsrchr(app_dir, L'\\');
    if (last_separator) *last_separator = 0;

    /* 2. the bundled Python */
    _snwprintf(python, PATH_CHARS - 1, L"%ls\\runtime\\pythonw.exe", app_dir);
    if (GetFileAttributesW(python) == INVALID_FILE_ATTRIBUTES) {
        show_error(
            L"The folder \"runtime\" was not found next to bacpipe.exe.\n"
            L"Please keep bacpipe.exe inside the bacpipe folder; to start it "
            L"from somewhere else, create a shortcut to it.",
            python);
        return 1;
    }

    /* 3. the command line */
    _snwprintf(command, PATH_CHARS - 1,
               L"\"%ls\" -E -s -X utf8 -m bacpipe_app %ls",
               python, arguments_after_program_name());

    /* tell the application where it is installed */
    SetEnvironmentVariableW(L"BACPIPE_APP_DIR", app_dir);

    /* 4. start Python and wait for it */
    STARTUPINFOW startup;
    PROCESS_INFORMATION process;
    ZeroMemory(&startup, sizeof(startup));
    ZeroMemory(&process, sizeof(process));
    startup.cb = sizeof(startup);

    if (!CreateProcessW(python, command, NULL, NULL, TRUE,
                        0, NULL, NULL, &startup, &process)) {
        wchar_t detail[256];
        _snwprintf(detail, 255, L"Windows error code %lu", GetLastError());
        show_error(L"bacpipe could not be started.", detail);
        return 1;
    }

    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return (int)exit_code;
}
