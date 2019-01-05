# haar-redactor

Blur faces in images using OpenCV's haar cascades.

## Installing

The module redact.py can be run directly or imported in to your own module. When run directly, the redacted output is display using an OpenCV window.

## Script usage

```
$ python redact.py -h
usage: redact.py [-h] (-i IMAGE | -v VIDEO | -w [WEBCAM]) [-x XML]

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Input image path
  -v VIDEO, --video VIDEO
                        Input video path
  -w [WEBCAM], --webcam [WEBCAM]
                        Input webcam number
  -x XML, --xml XML     Classifer xml path
```

Recently, The Rock (Dwayne Johnson) was awarded a Razzie for his role in Baywatch. 

https://www.independent.co.uk/arts-entertainment/films/news/dwayne-johnson-baywatch-razzie-rock-shit-sandwich-a8240786.html

[[https://github.com/deanwetherby/haar-face-redaction/blob/master/images/baywatch.jpg]]

Let's pretend that never happened.

```
$ python redact.py --image ./images/baywatch.jpg
```

[[]]


## Using Redactor as a module

The Redactor class can be imported and used directly in your own application. This will give you more control over things like saving the files as seen below.

```
from redact import Redactor

xml_path = "./haarcascade_frontalface_alt.xml"
image_path = "image.jpg"
output_path = "output.jpg"
img = cv2.imread(image_path)
redactor = Redactor(xml_path)
redacted_frame = redactor.redact_frame(img)
cv2.imwrite(output_path, redacted_frame) 

```
