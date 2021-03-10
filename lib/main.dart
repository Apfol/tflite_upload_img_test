import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:image/image.dart' as imgLib;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // Try running your application with "flutter run". You'll see the
        // application has a blue toolbar. Then, without quitting the app, try
        // changing the primarySwatch below to Colors.green and then invoke
        // "hot reload" (press "r" in the console where you ran "flutter run",
        // or simply save your changes to "hot reload" in a Flutter IDE).
        // Notice that the counter didn't reset back to zero; the application
        // is not restarted.
        primarySwatch: Colors.blue,
        // This makes the visual density adapt to the platform that you run
        // the app on. For desktop platforms, the controls will be smaller and
        // closer together (more dense) than on mobile platforms.
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {

  @override
  void initState() {
    Tflite.loadModel(
        model: "assets/liveness.tflite",
        labels: "assets/liveness.txt",
        numThreads: 1, // defaults to 1
        isAsset: true, // defaults to true, set to false to load resources outside assets
        useGpuDelegate: false // defaults to false, set to true to use GPU delegate
    );
    super.initState();
  }

  void _selectImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.getImage(source: ImageSource.gallery);
    //var imageBytes = (File(pickedFile.path).readAsBytesSync().buffer.asByteData()).buffer;
    //imgLib.Image image = imgLib.decodeImage(imageBytes.asUint8List());
    var image = imgLib.decodeImage(File(pickedFile.path).readAsBytesSync());
    var thumbnail = imgLib.copyResize(image, width: 32, height: 32);
    File(pickedFile.path).writeAsBytesSync(imgLib.encodePng(thumbnail));

    //useTFLitePackage(File(pickedFile.path));
    useTFLiteFlutterPackage(thumbnail, File(pickedFile.path));
    //usePlatformChannel(pickedFile.path);

  }

  void useTFLiteFlutterPackage(imgLib.Image image, File pickedImageFile) async {

    final interpreter = await tfl.Interpreter.fromAsset('liveness.tflite');
    TensorBuffer outputBuffer = TensorBuffer.createFixedSize(<int>[1, 2], TfLiteType.float32);
    TensorBuffer inputBuffer = TensorBuffer.createFixedSize(<int>[1, 32, 32, 3], TfLiteType.float32);
    inputBuffer.loadBuffer(imageToByteListFloat32(image, 32, 127, 255.0));

    TensorProcessor probabilityProcessor =
    TensorProcessorBuilder().add(DequantizeOp(0, 1 / 255.0)).build();
    TensorBuffer dequantizedBuffer = probabilityProcessor.process(outputBuffer);

    var _inputImage = TensorImage(TfLiteType.float32);
    _inputImage.loadImage(image);

    interpreter.run(inputBuffer.buffer, outputBuffer.buffer);

    List<String> labels = await FileUtil.loadLabels("assets/liveness.txt");

    /*TensorLabel tensorLabel = TensorLabel(labels, outputBuffer);
    );

    Map<String, double> doubleMap = tensorLabel.getMapWithFloatValue();*/

    print("TFLite Helper :: ${outputBuffer.getDoubleList()}");

  }

  void useTFLitePackage(File pickedImageFile) async {
    var recognitions = await Tflite.runModelOnImage(
        path: pickedImageFile.path,   // required
        imageMean: 0.0,   // defaults to 117.0
        imageStd: 255.0,  // defaults to 1.0
        numResults: 2,    // defaults to 5
        asynch: true      // defaults to true
    );
    print("TFLite Recognition :: ${recognitions.toString()}");
  }

  ByteBuffer imageToByteListFloat32(
      imgLib.Image image, int inputSize, double mean, double std) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (imgLib.getRed(pixel) - mean) / std;
        buffer[pixelIndex++] = (imgLib.getGreen(pixel) - mean) / std;
        buffer[pixelIndex++] = (imgLib.getBlue(pixel) - mean) / std;
      }
    }
    return convertedBytes.buffer;
  }

  Uint8List imageToByteListUint8(imgLib.Image image, int inputSize) {
    var convertedBytes = Uint8List(1 * inputSize * inputSize * 3);
    var buffer = Uint8List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = imgLib.getRed(pixel);
        buffer[pixelIndex++] = imgLib.getGreen(pixel);
        buffer[pixelIndex++] = imgLib.getBlue(pixel);
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              'Selecciona imagen',
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _selectImage,
        tooltip: 'Seleccionar imagen',
        child: Icon(Icons.add),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }

  void usePlatformChannel(String imagePath) async {
    const platform = const MethodChannel('samples.flutter.io/tensor');

    try {
      final String result = await platform.invokeMethod('doTensor', imagePath);
      print("Results :: " + result);
    } on PlatformException catch (e) {
      print("Failed to get result :: '${e.message}'.");
    }

  }
}
