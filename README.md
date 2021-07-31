# TensorRT inference server
Capstone project on the course <a href="https://otus.ru/lessons/cpp-professional/">"C++ Developer. Professional"</a>. <br/>
Asynchronous multithreaded server that loads pre-trained face detection model <a href="https://github.com/onnx/models/tree/master/vision/body_analysis/ultraface">UltraFace Onnx</a> into TensorRT inference engine and streams frames with detections using <a href="https://en.m.wikipedia.org/wiki/Motion_JPEG">Motion Jpeg</a> technology over HTTP.
