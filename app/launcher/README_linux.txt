bacpipe @VERSION@
==============

Start
-----
Run the file "bacpipe" in this folder (double-click it in your file manager
or run ./bacpipe in a terminal). A page opens in your web browser where you
choose your audio folder and the models, press "Run pipeline", and
afterwards explore the results in the dashboard.

Keep "bacpipe" inside this folder; to start it from somewhere else create a
symbolic link to it. No installation and no root rights are needed. To
remove bacpipe, delete this folder.


Where is my data?
-----------------
Results:            ~/bacpipe/bacpipe_results
Model checkpoints:  ~/bacpipe/bacpipe_model_checkpoints
Both folders can be changed on the "Settings" page. Your audio files are
only read, never changed.


If something does not work
--------------------------
* Start ./bacpipe in a terminal to see the messages of the program.
* Log files: ~/bacpipe/app_state
* Help: the "Help" button in the program, or
  https://github.com/bioacoustic-ai/bacpipe/issues
