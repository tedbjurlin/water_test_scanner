import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
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
  ColorDetectionResult({required this.colors, required this.exitCode});

  List<ColorOutput> colors;
  int exitCode;
}

typedef DetectColorsFunction = Pointer<NativeDetectorResult> Function(
    Pointer<Utf8> x, Pointer<Uint8> key, int width, int height);

typedef NativeDetectColorsFunction = Pointer<NativeDetectorResult> Function(
    Pointer<Utf8> x, Pointer<Uint8> key, Int32 width, Int32 height);

class ColorStripDetector {
  static Future<ColorDetectionResult> detectColors(
      String path, Uint8List ref, int width, int height) async {
    DynamicLibrary nativeColorDetection = _getDynamicLibrary();

    print(nativeColorDetection.providesSymbol("native_detect_colors"));
    print(nativeColorDetection.toString());

    final detectColors = nativeColorDetection.lookupFunction<
        NativeDetectColorsFunction,
        DetectColorsFunction>("native_detect_colors");

    final Pointer<Uint8> pointer =
        malloc.allocate<Uint8>(ref.buffer.lengthInBytes);
    final pointerList = pointer.asTypedList(ref.buffer.lengthInBytes);
    pointerList.setAll(0, ref);

    NativeDetectorResult detectionResult =
        detectColors(path.toNativeUtf8(), pointer, width, height).ref;

    print(detectionResult.exitCode);
    print(detectionResult);

    malloc.free(pointer);

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
    ], exitCode: detectionResult.exitCode);
  }

  static DynamicLibrary _getDynamicLibrary() {
    final DynamicLibrary nativeEdgeDetection = Platform.isAndroid
        ? DynamicLibrary.open("libnative_water_test_scanning.so")
        : DynamicLibrary.process();
    return nativeEdgeDetection;
  }
}
