Track'n Label
=============

1. Download latest dlib:
------------------------
http://dlib.net/


2. copy meta_editor.h and meta_editor.cpp from Trouserdata/ into dlib-19.8/tools/imglab/src
-------------------------------------------------------------------------------------------

3. compile imglab
-----------------
cd dlib-19.8/tools/imglab
mkdir build
cd build
cmake ..
make -j<cores>
sudo make install # to make imglab accessible as a executable


3. label your dataset, e.g. photos_1
---------------------
cd photos_1
imglab -c path-to-photos_1/mydataset.xml path-to-photos_1 # this will create the .xml where the labels will be stored
imglab ~/path-to-photos_1/mydataset.xml

a. enter 't_' (without quotes) in the Next label box
b. Hold shift to draw the bounding rectangle on the trousers
c. Arrow down to go through the images, it should keep the rectangle on the object
d. In case the current rectangle cannot track the object well enough anymore:
	- double click on the bounding box
	- Delete by pressing backspace key
	- go to a.



5.Finally to save all the tracked and labeled images
----------------------------------------------------
File -> Save # it will save the labels in ~/path-to-photos_1/mydataset.xml



