import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:path_provider/path_provider.dart';

// This class corresponds to the ColorOutput struct in the C++ code. It acts as
// a go-between for the C++ ColorOutput struct and the Dart ColorOutput class.
// If this is used directly instead of copying the values to a normal dart class,
// memory leaks are common.
final class NativeColorOutput extends Struct {
  @Int32()
  external int idx;

  @Int32()
  external int red;

  @Int32()
  external int green;

  @Int32()
  external int blue;

  @Double()
  external double value;

  factory NativeColorOutput.allocate(
          int idx, int red, int green, int blue, double value) =>
      calloc<NativeColorOutput>().ref
        ..idx = idx
        ..red = red
        ..green = green
        ..blue = blue
        ..value = value;

  @override
  String toString() {
    return "{"
        "idx = $idx"
        "red = $red"
        "green = $green"
        "blue = $blue"
        "value = $value"
        "}";
  }
}

// This is the dart version of the ColorOutput class.
class ColorOutput {
  ColorOutput(
      {required this.idx,
      required this.red,
      required this.green,
      required this.blue,
      required this.value});

  int idx;

  int red;

  int green;

  int blue;

  double value;
}

// This converts a Dart:ffi NativeColorOutput to a Dart ColorOutput class.
ColorOutput fromNativeColorOutput(NativeColorOutput out) {
  return ColorOutput(
      idx: out.idx,
      red: out.red,
      green: out.green,
      blue: out.blue,
      value: out.value);
}

// The NativeDetectorResult serves the same purpose as the NativeColorOutput class,
// but for the ColorDetectionResult class.
final class NativeDetectorResult extends Struct {
  external Pointer<NativeColorOutput> color1;
  external Pointer<NativeColorOutput> color2;
  external Pointer<NativeColorOutput> color3;
  external Pointer<NativeColorOutput> color4;
  external Pointer<NativeColorOutput> color5;
  external Pointer<NativeColorOutput> color6;
  external Pointer<NativeColorOutput> color7;
  external Pointer<NativeColorOutput> color8;
  external Pointer<NativeColorOutput> color9;
  external Pointer<NativeColorOutput> color10;
  external Pointer<NativeColorOutput> color11;
  external Pointer<NativeColorOutput> color12;
  external Pointer<NativeColorOutput> color13;
  external Pointer<NativeColorOutput> color14;
  external Pointer<NativeColorOutput> color15;
  external Pointer<NativeColorOutput> color16;

  @Int32()
  external int size;

  @Int32()
  external int exitCode;

  factory NativeDetectorResult.allocate(
          Pointer<NativeColorOutput> color1,
          Pointer<NativeColorOutput> color2,
          Pointer<NativeColorOutput> color3,
          Pointer<NativeColorOutput> color4,
          Pointer<NativeColorOutput> color5,
          Pointer<NativeColorOutput> color6,
          Pointer<NativeColorOutput> color7,
          Pointer<NativeColorOutput> color8,
          Pointer<NativeColorOutput> color9,
          Pointer<NativeColorOutput> color10,
          Pointer<NativeColorOutput> color11,
          Pointer<NativeColorOutput> color12,
          Pointer<NativeColorOutput> color13,
          Pointer<NativeColorOutput> color14,
          Pointer<NativeColorOutput> color15,
          Pointer<NativeColorOutput> color16,
          int size,
          int exitCode) =>
      calloc<NativeDetectorResult>().ref
        ..color1 = color1
        ..color2 = color2
        ..color3 = color3
        ..color4 = color4
        ..color5 = color5
        ..color6 = color6
        ..color7 = color7
        ..color8 = color8
        ..color9 = color9
        ..color10 = color10
        ..color11 = color11
        ..color12 = color12
        ..color13 = color13
        ..color14 = color14
        ..color15 = color15
        ..color16 = color16
        ..size = size
        ..exitCode = exitCode;

  @override
  String toString() {
    return "{"
        "  color1 = ${color1.ref.toString()},"
        "  color2 = ${color2.ref.toString()},"
        "  color3 = ${color3.ref.toString()},"
        "  color4 = ${color4.ref.toString()},"
        "  color5 = ${color5.ref.toString()},"
        "  color6 = ${color6.ref.toString()},"
        "  color7 = ${color7.ref.toString()},"
        "  color8 = ${color8.ref.toString()},"
        "  color9 = ${color9.ref.toString()},"
        "  color10 = ${color10.ref.toString()},"
        "  color11 = ${color11.ref.toString()},"
        "  color12 = ${color12.ref.toString()},"
        "  color13 = ${color13.ref.toString()},"
        "  color14 = ${color14.ref.toString()},"
        "  color15 = ${color15.ref.toString()},"
        "  color16 = ${color16.ref.toString()},"
        "  exitCode = $exitCode"
        "}";
  }
}

class ColorDetectionResult {
  ColorDetectionResult(
      {required this.colors, required this.image, required this.imFile, required this.exitCode});

  List<ColorOutput> colors;
  Image image;
  File imFile;
  int exitCode;

  Map<String, dynamic> toFirebaseRecord(
    String imageLink, dynamic loc, String waterType, DateTime timestamp){
    return {
      "image": imageLink,
      "timestamp": timestamp.microsecondsSinceEpoch,
      "location": loc,
      "Water Type": waterType,
      "pH": colors[0].value,
      "Hardness": colors[1].value,
      "Hydrogen Sulfide": colors[2].value,
      "Iron": colors[3].value,
      "Copper": colors[4].value,
      "Lead": colors[5].value,
      "Manganese": colors[6].value,
      "Total Chlorine": colors[7].value,
      "Mercury": colors[8].value,
      "Nitrate": colors[9].value,
      "Nitrite": colors[10].value,
      "Sulfate": colors[11].value,
      "Zinc": colors[12].value,
      "Flouride": colors[13].value,
      "Sodium Chloride": colors[14].value,
      "Total Alkalinity": colors[15].value
    };
  }
}

// The expected type of the C++ class, in Dart types
typedef DetectColorsFunction = Pointer<NativeDetectorResult> Function(
    Pointer<Utf8> x,
    Pointer<Uint8> key,
    int length,
    Pointer<Pointer<Uint8>> encodedImage);

// The expected type of the C++ class, in Dar:ffi Native types
typedef NativeDetectColorsFunction = Pointer<NativeDetectorResult> Function(
    Pointer<Utf8> x,
    Pointer<Uint8> key,
    Int32 length,
    Pointer<Pointer<Uint8>> encodedImage);

// The class that manages the color detector.
class ColorStripDetector {
  // The dart version of the detect_colors function. It takes the path to the image
  // from the camera, and a UInt8List of the color card refernce bytes.
  static Future<ColorDetectionResult> detectColors(
      String path, Uint8List ref) async {
    // first we get the dynamic library of our C++ code.
    DynamicLibrary nativeColorDetection = _getDynamicLibrary();

    // These are checks to make sure that the funciton is found.
    debugPrint(
        nativeColorDetection.providesSymbol("native_detect_colors").toString());
    debugPrint(nativeColorDetection.toString());

    // Here we lookup the C++ function and convert it to a Dart function.
    final detectColors = nativeColorDetection.lookupFunction<
        NativeDetectColorsFunction,
        DetectColorsFunction>("native_detect_colors");

    // We allocate an array of unsigned 8-bit integers to store the bytes of our
    // reference image, and store the bytelist in it.
    final Pointer<Uint8> pointer =
        malloc.allocate<Uint8>(ref.buffer.lengthInBytes);
    final pointerList = pointer.asTypedList(ref.buffer.lengthInBytes);
    pointerList.setAll(0, ref);

    // We allocate 8 bytes of memory with a pointer to our outputImage, and take a
    // pointer to that allocated memory.
    Pointer<Pointer<Uint8>> encodedImPtr = malloc.allocate(8);

    // detectColors is called with our allocated memory
    NativeDetectorResult detectionResult = detectColors(
            path.toNativeUtf8(), pointer, ref.lengthInBytes, encodedImPtr)
        .ref;

    debugPrint(detectionResult.exitCode.toString());
    debugPrint(detectionResult.toString());

    // We can now retrieve the outputImage from memory and store it in a list of bytes.
    Pointer<Uint8> cppPointer = encodedImPtr.elementAt(0).value;
    Uint8List imageBytes = cppPointer.asTypedList(detectionResult.size);

    // Copying that bytelist to a seperate bytelist decouples it from the C++ allocated
    // memory. The prevents segmentation faults when C++ frees the memory.
    Uint8List newImage = Uint8List(imageBytes.length);
    for (int i = 0; i < imageBytes.length; i++) {
      newImage[i] = imageBytes[i];
    }
    // The image is finally decoded into a flutter Image widget and the remaining pointers are freed.
    Image image = Image.memory(newImage);

    Directory tempDir = await getTemporaryDirectory();
    File imFile = await File('${tempDir.path}/image.jpg').create();
    await imFile.writeAsBytes(newImage);

    malloc.free(pointer);
    malloc.free(cppPointer);
    malloc.free(encodedImPtr);

    return ColorDetectionResult(colors: [
      fromNativeColorOutput(detectionResult.color1.ref),
      fromNativeColorOutput(detectionResult.color2.ref),
      fromNativeColorOutput(detectionResult.color3.ref),
      fromNativeColorOutput(detectionResult.color4.ref),
      fromNativeColorOutput(detectionResult.color5.ref),
      fromNativeColorOutput(detectionResult.color6.ref),
      fromNativeColorOutput(detectionResult.color7.ref),
      fromNativeColorOutput(detectionResult.color8.ref),
      fromNativeColorOutput(detectionResult.color9.ref),
      fromNativeColorOutput(detectionResult.color10.ref),
      fromNativeColorOutput(detectionResult.color11.ref),
      fromNativeColorOutput(detectionResult.color12.ref),
      fromNativeColorOutput(detectionResult.color13.ref),
      fromNativeColorOutput(detectionResult.color14.ref),
      fromNativeColorOutput(detectionResult.color15.ref),
      fromNativeColorOutput(detectionResult.color16.ref),
    ], image: image, imFile: imFile, exitCode: detectionResult.exitCode);
  }

  static DynamicLibrary _getDynamicLibrary() {
    final DynamicLibrary nativeEdgeDetection = Platform.isAndroid
        ? DynamicLibrary.open("libnative_water_test_scanning.so")
        : DynamicLibrary.process();
    return nativeEdgeDetection;
  }
}
