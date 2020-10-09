package org.tensorflow.lite.examples.classification;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class DepthEstimationModel {
    protected Interpreter tflite;
    private static final Logger LOGGER = new Logger();

    protected DepthEstimationModel(Activity activity) throws IOException {
        MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(activity, getModelPath());

        Interpreter.Options tfliteOptions = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();

        if(compatList.isDelegateSupportedOnThisDevice()){
            // if the device has a supported GPU, add the GPU delegate
            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
            tfliteOptions.addDelegate(gpuDelegate);
        } else {
            // if the GPU is not supported, run on 4 threads
            tfliteOptions.setNumThreads(4);
        }

        tflite = new Interpreter(modelBuffer, tfliteOptions);
    }

    protected String getModelPath() {
        return "tflite_pydnet++.tflite";
    }

    public void close() {
        tflite.close();
        tflite = null;
    }

    public float[] inference(float[][][][] input, int height, int width) {
        float[][][][] tmpOutput = new float[1][height][width][1];
        long startTime = SystemClock.uptimeMillis();
        tflite.run(input, tmpOutput);
        long inferenceTime = SystemClock.uptimeMillis() - startTime;
        LOGGER.d("inference time %dms", inferenceTime);

        float[] output = new float[width * height];
        for (int row=0; row < height; row++){
            for (int col=0; col < width; col++) {
                output[row * width + col] = tmpOutput[0][row][col][0];
            }
        }
        return output;
    }

    public static float[][][][] getPixelFromBitmap(Bitmap frame){
        int[] pixels = new int[frame.getWidth() * frame.getHeight()];
        frame.getPixels(pixels, 0, frame.getWidth(), 0, 0, frame.getWidth(), frame.getHeight());

        float[][][][] output = new float[1][frame.getHeight()][frame.getWidth()][3];
        int y = 0, x = 0, c = 0;
        float min = 10000, max = 0;
        for (int pixel : pixels) {
            output[0][y][x][0] = Color.red(pixel) / (float)255.;
            output[0][y][x][1] = Color.green(pixel) / (float)255.;
            output[0][y][x][2] = Color.blue(pixel) / (float)255.;
            x += 1;
            c += 1;
            if (x == frame.getWidth()) {
                y += 1;
                x = 0;
            }
        }
        return output;
    }

    public void inferenceBitmap(Bitmap inputBitmap, Bitmap outputBitmap) {
        int width = 640, height = 448;
        Bitmap resizedBitmap = Bitmap.createBitmap(inputBitmap, 0, 0, width, height);
        float[][][][] input = getPixelFromBitmap(resizedBitmap);

//        long startTime = SystemClock.uptimeMillis();
        float[] output = inference(input, height, width);
//        long inferenceTime = SystemClock.uptimeMillis() - startTime;
//        LOGGER.d("inference time %dms", inferenceTime);
        int[] colors = applyColorMap(output);
        outputBitmap.setPixels(colors, 0, width, 0, 0, width, height);
    }

    private int[] applyColorMap(float[] output) {
        int[] color = new int[output.length];
        for (int i=0; i < output.length; i++) {
            int x = (int) Math.max(Math.min(256 - output[i] * 10, 255), 0);
            color[i] = 0xff << 24 | (x & 0xff) << 16 | (x & 0xff) << 8 | (x & 0xff);
        }
        return color;
    }
}
