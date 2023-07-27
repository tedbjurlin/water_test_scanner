import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';

final class ColorOutput extends Struct {
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

  factory ColorOutput.allocate(
          int idx, int red, int green, int blue, double value) =>
      calloc<ColorOutput>().ref
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

final class NativeDetectorResult extends Struct {
  external Pointer<ColorOutput> color1;
  external Pointer<ColorOutput> color2;
  external Pointer<ColorOutput> color3;
  external Pointer<ColorOutput> color4;
  external Pointer<ColorOutput> color5;
  external Pointer<ColorOutput> color6;
  external Pointer<ColorOutput> color7;
  external Pointer<ColorOutput> color8;
  external Pointer<ColorOutput> color9;
  external Pointer<ColorOutput> color10;
  external Pointer<ColorOutput> color11;
  external Pointer<ColorOutput> color12;
  external Pointer<ColorOutput> color13;
  external Pointer<ColorOutput> color14;
  external Pointer<ColorOutput> color15;
  external Pointer<ColorOutput> color16;

  @Int32()
  external int exitCode;

  factory NativeDetectorResult.allocate(
          Pointer<ColorOutput> color1,
          Pointer<ColorOutput> color2,
          Pointer<ColorOutput> color3,
          Pointer<ColorOutput> color4,
          Pointer<ColorOutput> color5,
          Pointer<ColorOutput> color6,
          Pointer<ColorOutput> color7,
          Pointer<ColorOutput> color8,
          Pointer<ColorOutput> color9,
          Pointer<ColorOutput> color10,
          Pointer<ColorOutput> color11,
          Pointer<ColorOutput> color12,
          Pointer<ColorOutput> color13,
          Pointer<ColorOutput> color14,
          Pointer<ColorOutput> color15,
          Pointer<ColorOutput> color16,
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
    Pointer<Utf8> x);

class ColorStripDetector {
  static Future<ColorDetectionResult> detectColors(String path) async {
    DynamicLibrary nativeColorDetection = _getDynamicLibrary();

    print(nativeColorDetection.providesSymbol("native_detect_colors"));
    print(nativeColorDetection.toString());

    final detectColors = nativeColorDetection.lookupFunction<
        DetectColorsFunction, DetectColorsFunction>("native_detect_colors");

    NativeDetectorResult detectionResult =
        detectColors(path.toNativeUtf8()).ref;

    print(detectionResult.exitCode);
    print(detectionResult);

    return ColorDetectionResult(colors: [
      detectionResult.color1.ref,
      detectionResult.color2.ref,
      detectionResult.color3.ref,
      detectionResult.color4.ref,
      detectionResult.color5.ref,
      detectionResult.color6.ref,
      detectionResult.color7.ref,
      detectionResult.color8.ref,
      detectionResult.color9.ref,
      detectionResult.color10.ref,
      detectionResult.color11.ref,
      detectionResult.color12.ref,
      detectionResult.color13.ref,
      detectionResult.color14.ref,
      detectionResult.color15.ref,
      detectionResult.color16.ref,
    ], exitCode: detectionResult.exitCode);
  }

  static DynamicLibrary _getDynamicLibrary() {
    final DynamicLibrary nativeEdgeDetection = Platform.isAndroid
        ? DynamicLibrary.open("libnative_water_test_scanning.so")
        : DynamicLibrary.process();
    return nativeEdgeDetection;
  }
}
