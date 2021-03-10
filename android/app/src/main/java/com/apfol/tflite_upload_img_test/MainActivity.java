package com.apfol.tflite_upload_img_test;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.PersistableBundle;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.apfol.tflite_upload_img_test.ml.Liveness;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Map;

import io.flutter.embedding.android.FlutterActivity;
import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugins.GeneratedPluginRegistrant;

public class MainActivity extends FlutterActivity {

    private static final String CHANNEL = "samples.flutter.io/tensor";

    @Override
    public void configureFlutterEngine(@NonNull FlutterEngine flutterEngine) {
        GeneratedPluginRegistrant.registerWith(flutterEngine);
        new MethodChannel(getFlutterEngine().getDartExecutor().getBinaryMessenger(), CHANNEL).setMethodCallHandler(
                (call, result) -> {
                    if (call.method.equals("doTensor")) {
                        // Get ByteBuffer Image from file path
                        ByteBuffer byteBuffer = getImageByteBuffer(new File(call.arguments.toString()));
                        startTensorFlowLite(byteBuffer);
                        result.success(call.arguments);
                    } else {
                        result.notImplemented();
                    }
                });
    }

    //Get ByteBuffer to TFL model input float32 [1 32 32 3]
    private ByteBuffer getImageByteBuffer(File imageFile) {
        String filePath = imageFile.getPath();
        Bitmap bitmap = BitmapFactory.decodeFile(filePath);
        bitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true);
        ByteBuffer input = ByteBuffer.allocateDirect(32 * 32 * 3 * 4).order(ByteOrder.nativeOrder());
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int px = bitmap.getPixel(x, y);

                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);

                // Normalize channel values to [-1.0, 1.0]. This requirement depends
                // on the model. For example, some models might require values to be
                // normalized to the range [0.0, 1.0] instead.
                float rf = (r - 127) / 255.0f;
                float gf = (g - 127) / 255.0f;
                float bf = (b - 127) / 255.0f;

                input.putFloat(rf);
                input.putFloat(gf);
                input.putFloat(bf);
            }
        }
        return input;
    }

    private void startTensorFlowLite(ByteBuffer byteBuffer) {
        try {
            Liveness model = Liveness.newInstance(this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Liveness.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();



            final String ASSOCIATED_AXIS_LABELS = "liveness.txt";
            List<String> associatedAxisLabels = null;

            try {
                associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
            } catch (IOException e) {
                Log.e("tfliteSupport", "Error reading label file", e);
            }


            // Post-processor which dequantize the result
            TensorProcessor probabilityProcessor =
                    new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();

            if (null != associatedAxisLabels) {
                // Map of labels and their corresponding probability
                TensorLabel labels = new TensorLabel(associatedAxisLabels,
                        probabilityProcessor.process(outputFeature0));

                // Create a map to access the result based on label
                Map<String, Float> floatMap = labels.getMapWithFloatValue();
                System.out.println("Tensor Flow Results :: " + floatMap.toString());
            }

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }


}
