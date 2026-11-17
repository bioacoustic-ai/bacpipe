bacpipe @VERSION@
==============

Start
-----
Double-click bacpipe.exe. A page opens in your web browser where you choose
your audio folder and the models, press "Run pipeline", and afterwards
explore the results in the dashboard.

Keep bacpipe.exe inside this folder. To start it from somewhere else
(desktop, start menu), right-click bacpipe.exe and create a shortcut.

No installation and no administrator rights are needed. To remove bacpipe,
delete this folder.


If Windows shows "Windows protected your PC"
--------------------------------------------
bacpipe is free, open source software and its program file is not signed
with a paid certificate, so Windows does not know it yet.
Click "More info" and then "Run anyway".

To avoid the message completely: before unpacking the downloaded .zip file,
right-click it, choose "Properties", tick "Unblock" and click OK.

If your organisation does not allow bacpipe.exe to run at all, try
bacpipe_debug.cmd in this folder. It starts the same program in a console
window.


Where is my data?
-----------------
Results:            C:\Users\<you>\bacpipe\bacpipe_results
Model checkpoints:  C:\Users\<you>\bacpipe\bacpipe_model_checkpoints
Both folders can be changed on the "Settings" page. Your audio files are
only read, never changed.


If something does not work
--------------------------
* Start bacpipe_debug.cmd to see the messages of the program.
* Log files: C:\Users\<you>\bacpipe\app_state
* Help: the "Help" button in the program, or
  https://github.com/bioacoustic-ai/bacpipe/issues
* If files are reported missing after unpacking, unpack the .zip to a
  location with a short path, for example C:\Users\<you>\bacpipe_app
