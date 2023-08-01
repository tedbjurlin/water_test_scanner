import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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

ColorOutput fromNativeColorOutput(NativeColorOutput out) {
  return ColorOutput(
      idx: out.idx,
      red: out.red,
      green: out.green,
      blue: out.blue,
      value: out.value);
}

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
  external int width;

  @Int32()
  external int height;

  @Int32()
  external int buffer_size;

  external Pointer<Uint8> image;

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
          int width,
          int height,
          int buffer_size,
          Pointer<Uint8> image,
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
        ..width = width
        ..height = height
        ..buffer_size = buffer_size
        ..image = image
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
        "  width = $width,"
        "  height = $height,"
        "  buffer_size = $buffer_size"
        "  exitCode = $exitCode"
        "}";
  }
}

class ColorDetectionResult {
  ColorDetectionResult(
      {required this.colors,
      required this.width,
      required this.height,
      required this.image,
      required this.exitCode});

  List<ColorOutput> colors;

  int width;
  int height;

  Image image;

  int exitCode;
}

typedef DetectColorsFunction = Pointer<NativeDetectorResult> Function(
    Pointer<Uint8> x,
    Pointer<Int32> xw,
    Pointer<Int32> xh,
    Pointer<Uint8> y,
    Pointer<Int32> w,
    Pointer<Int32> h,
    Pointer<Bool> isYUV);

typedef NativeEncodeImFunction = Int32 Function(Int32 width, Int32 height,
    Pointer<Uint8> bytes, Pointer<Pointer<Uint8>> encodedOutput);

typedef EncodeImFunction = int Function(int width, int height,
    Pointer<Uint8> bytes, Pointer<Pointer<Uint8>> encodedOutput);

class ColorStripDetector {
  static Future<ColorDetectionResult> detectColors(
      Uint8List imageList,
      int imageWidth,
      int imageHeight,
      Uint8List list,
      int width,
      int height,
      bool isYUV) async {
    DynamicLibrary nativeColorDetection = _getDynamicLibrary();

    final detectColors = nativeColorDetection.lookupFunction<
        DetectColorsFunction, DetectColorsFunction>("native_detect_colors");

    final encodeIm = nativeColorDetection
        .lookupFunction<NativeEncodeImFunction, EncodeImFunction>("encodeIm");

    print(imageWidth);
    print(imageHeight);

    final Pointer<Uint8> imagePointer =
        malloc.allocate<Uint8>(imageList.buffer.lengthInBytes);
    final imagePointerList =
        imagePointer.asTypedList(imageList.buffer.lengthInBytes);
    imagePointerList.setAll(0, imageList);

    final iwPointer = calloc<Int32>();
    iwPointer.value = imageWidth;

    final ihPointer = calloc<Int32>();
    ihPointer.value = imageHeight;

    final Pointer<Uint8> pointer =
        malloc.allocate<Uint8>(list.buffer.lengthInBytes);
    final pointerList = pointer.asTypedList(list.buffer.lengthInBytes);
    pointerList.setAll(0, list);

    final wPointer = calloc<Int32>();
    wPointer.value = width;

    final hPointer = calloc<Int32>();
    hPointer.value = height;

    final bPointer = calloc<Bool>();
    bPointer.value = isYUV;

    // NativeDetectorResult detectionResult = detectColors(imagePointer, iwPointer,
    //         ihPointer, pointer, wPointer, hPointer, bPointer)
    //     .ref;

    // print(detectionResult.exitCode);

    // print(detectionResult.width);
    // print(detectionResult.height);
    // print(detectionResult.buffer_size);
    // print(detectionResult.height * detectionResult.width);

    Pointer<Pointer<Uint8>> encodedImage = malloc.allocate(8);

    int encodedBufferSize =
        encodeIm(imageWidth, imageHeight, imagePointer, encodedImage);

    Pointer<Uint8> cppPointer = encodedImage.elementAt(0).value;
    Uint8List encodedImBytes = cppPointer.asTypedList(encodedBufferSize);

    print(encodedImBytes.lengthInBytes);
    print(encodedImBytes);
    print(encodedBufferSize);

    Image myImg = Image.memory(encodedImBytes,
        height: imageHeight.toDouble(), width: imageWidth.toDouble());

    ColorDetectionResult out = ColorDetectionResult(colors: [
      // fromNativeColorOutput(detectionResult.color1.ref),
      // fromNativeColorOutput(detectionResult.color2.ref),
      // fromNativeColorOutput(detectionResult.color3.ref),
      // fromNativeColorOutput(detectionResult.color4.ref),
      // fromNativeColorOutput(detectionResult.color5.ref),
      // fromNativeColorOutput(detectionResult.color6.ref),
      // fromNativeColorOutput(detectionResult.color7.ref),
      // fromNativeColorOutput(detectionResult.color8.ref),
      // fromNativeColorOutput(detectionResult.color9.ref),
      // fromNativeColorOutput(detectionResult.color10.ref),
      // fromNativeColorOutput(detectionResult.color11.ref),
      // fromNativeColorOutput(detectionResult.color12.ref),
      // fromNativeColorOutput(detectionResult.color13.ref),
      // fromNativeColorOutput(detectionResult.color14.ref),
      // fromNativeColorOutput(detectionResult.color15.ref),
      // fromNativeColorOutput(detectionResult.color16.ref),
    ], width: width, height: height, image: myImg, exitCode: -1);

    malloc.free(imagePointer);
    malloc.free(iwPointer);
    malloc.free(ihPointer);
    malloc.free(pointer);
    malloc.free(wPointer);
    malloc.free(hPointer);
    malloc.free(bPointer);
    malloc.free(cppPointer);
    malloc.free(encodedImage);

    return out;
  }

  static DynamicLibrary _getDynamicLibrary() {
    final DynamicLibrary nativeEdgeDetection = Platform.isAndroid
        ? DynamicLibrary.open("libnative_water_test_scanning.so")
        : DynamicLibrary.process();
    return nativeEdgeDetection;
  }
}

// figure out how to dereference color outputs

// fix length error causing frame freezing