bacpipe @VERSION@
==============

Start
-----
Double-click bacpipe.app. A page opens in your web browser where you choose
your audio folder and the models, press "Run pipeline", and afterwards
explore the results in the dashboard.

The first time, macOS may say that bacpipe "cannot be opened because the
developer cannot be verified": bacpipe is free, open source software and is
not signed with a paid certificate. Right-click (or Control-click)
bacpipe.app, choose "Open" and confirm with "Open". Alternatively run this
once in the Terminal:

    xattr -dr com.apple.quarantine /path/to/bacpipe.app

No installation and no administrator rights are needed. To remove bacpipe,
delete bacpipe.app.


Where is my data?
-----------------
Results:            ~/bacpipe/bacpipe_results
Model checkpoints:  ~/bacpipe/bacpipe_model_checkpoints
Both folders can be changed on the "Settings" page. Your audio files are
only read, never changed.


If something does not work
--------------------------
* Start it in the Terminal to see the messages of the program:
      /path/to/bacpipe.app/Contents/MacOS/bacpipe
* Log files: ~/bacpipe/app_state
* Help: the "Help" button in the program, or
  https://github.com/bioacoustic-ai/bacpipe/issues
