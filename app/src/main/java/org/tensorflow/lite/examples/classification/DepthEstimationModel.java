package org.tensorflow.lite.examples.classification;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class DepthEstimationModel {
    protected Interpreter tflite;

    protected DepthEstimationModel(Activity activity) throws IOException {
        MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(activity, getModelPath());

        Interpreter.Options tfliteOptions = new Interpreter.Options();
        GpuDelegate gpuDelegate = new GpuDelegate();
        tfliteOptions.addDelegate(gpuDelegate);
        tfliteOptions.setNumThreads(4);

        tflite = new Interpreter(modelBuffer, tfliteOptions);
    }

    protected String getModelPath() {
        return "pydnet++.tflite";
    }

    public void close() {
        tflite.close();
        tflite = null;
    }

    public float[] inference(float[] input, int height, int width) {
        float[][] tmpOutput = new float[height][width];
        tflite.run(input, tmpOutput);

        float[] output = new float[width * height];
        for (int row=0; row < height; row++){
            System.arraycopy(tmpOutput[row], 0, output, row * width, width);
        }
        return output;
    }

    public static float[] getPixelFromBitmap(Bitmap frame){
        int numberOfPixels = frame.getWidth() * frame.getHeight()*3;
        int[] pixels = new int[frame.getWidth() * frame.getHeight()];
        frame.getPixels(pixels, 0, frame.getWidth(), 0, 0, frame.getWidth(), frame.getHeight());

        float[] output = new float[numberOfPixels];
        int i = 0;
        for (int pixel : pixels) {
            output[i * 3] = Color.red(pixel) / (float)255.;
            output[i * 3 + 1] = Color.green(pixel) / (float)255.;
            output[i * 3 + 2] = Color.blue(pixel) / (float)255.;
            i += 1;
        }
        return output;
    }

    public void inferenceBitmap(Bitmap inputBitmap, Bitmap outputBitmap) {
        float[] input = getPixelFromBitmap(inputBitmap);
        float[] output = inference(input, inputBitmap.getHeight(), inputBitmap.getWidth());
        int[] colors = applyColorMap(output);
        outputBitmap.setPixels(colors, 0, inputBitmap.getWidth(), 0, 0, inputBitmap.getWidth(), inputBitmap.getHeight());
    }

    private int[] applyColorMap(float[] output) {
        int[] color = new int[output.length];
        for (int i=0; i < output.length; i++) {
            float x = output[i];
            int red = (int) (-0.0045 * Math.pow(x, 2) + 0.2337 * x + 248.25);
            int green = (int) (2e-5 * Math.pow(x, 3) - 0.0023 * Math.pow(x, 2) + 1.3374 * x + 243.68);
            int blue = (int) (-4e-5 * Math.pow(x, 3) + 0.0112 * Math.pow(x, 2) - 0.1947 * x + 37.514);
            red = Math.max(Math.min(red, 255), 0);
            green = Math.max(Math.min(green, 255), 0);
            blue = Math.max(Math.min(blue, 255), 0);
            color[i] = (red & 0xff) << 16 | (green & 0xff) << 8 | (blue & 0xff);
        }
        return color;
    }
}
