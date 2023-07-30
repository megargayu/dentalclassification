package com.ayush.dentalclassification;

import static com.ayush.dentalclassification.MainActivity.assetFilePath;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Objects;
import java.util.Random;

public class ModelManager {
    static class Model {
        String path;
        Module module;

        public Model(String path, Module module) {
            this.path = path;
            this.module = module;
        }
    }

    final Random random = new Random();

    final AppCompatActivity activity;
    public final HashMap<String, Model> models = new HashMap<>();

    final ArrayList<Tensor> images = new ArrayList<>();
    final ArrayList<Float> classifications = new ArrayList<>();

    public ModelManager(AppCompatActivity activity) throws IOException {
        this.activity = activity;

        String[] modelsLst = this.activity.getResources().getStringArray(R.array.models);
        String[] modelPaths = this.activity.getResources().getStringArray(R.array.model_paths);

        for (int i = 0; i < modelsLst.length; i++) {
            models.put(modelsLst[i], new Model(modelPaths[i], null));
        }
    }

    public void loadAllImages() throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(activity.getAssets().open("tensors.txt")));

        float[] arr = new float[3 * 224 * 224];
        String line;

        while ((line = reader.readLine()) != null && line.length() != 0) {
            String[] values = line.split(", ");
            assert values.length - 1 == arr.length;

            Float label = Float.parseFloat(values[0]);

            for (int i = 1; i < values.length; i++) {
                arr[i - 1] = Float.parseFloat(values[i]);
            }

            images.add(Tensor.fromBlob(arr, new long[]{1, 3, 224, 224}));
            classifications.add(label);
        }
    }

    public static class RandomImageResult {
        Bitmap image;
        Float groundTruth;
        Float output;

        public RandomImageResult(Bitmap image, Float groundTruth, Float output) {
            this.image = image;
            this.groundTruth = groundTruth;
            this.output = output;
        }
    }

    public RandomImageResult runRandomImage(String model) throws IOException {
        int randomIndex = random.nextInt(images.size());

        Tensor imageTensor = images.get(randomIndex);
        Float groundTruth = classifications.get(randomIndex);

        float output = runModel(model, imageTensor);

        Bitmap outBitmap = rgbTensorToBitmap(imageTensor, 224, 224,
                new float[]{0.0f, 0.0f, 0.0f}, new float[]{1.0f, 1.0f, 1.0f});

        return new RandomImageResult(outBitmap, groundTruth, output);
    }

    private Bitmap rgbTensorToBitmap(Tensor tensor, int width, int height,
                                     float[] normMeanRGB, float[] normStdRGB) {
        float[] tensorArr = tensor.getDataAsFloatArray();

        Bitmap output = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        int pixelsCount = width * height;
        int[] pixels = new int[pixelsCount];

        for (int i = 0; i < pixelsCount; i++) {
            int r = Math.round(tensorArr[i] * 255.0f * normStdRGB[0] + normMeanRGB[0]);
            int g = Math.round(tensorArr[i + pixelsCount] * 255.0f * normStdRGB[1] + normMeanRGB[1]);
            int b = Math.round(tensorArr[i + pixelsCount * 2] * 255.0f * normStdRGB[2] + normMeanRGB[2]);
            pixels[i] = (255 << 24) | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
        }

        output.setPixels(pixels, 0, width, 0, 0, width, height);

        return output;
    }

    public float runFullTest(String model) throws IOException {
        int numCorrect = 0;

        for (int i = 0; i < images.size(); i++) {
            Tensor imageTensor = images.get(i);
            Float groundTruth = classifications.get(i);

            float output = runModel(model, imageTensor);
            numCorrect += (output == groundTruth) ? 1 : 0;
        }

        return ((float) numCorrect) / ((float) images.size());
    }

    public Module loadModule(String modelPath) throws IOException {
        return LiteModuleLoader.load(assetFilePath(this.activity, modelPath));
    }

    private float runModel(String model, Tensor input) {
        Module module = Objects.requireNonNull(models.get(model)).module;
        final Tensor output = module.forward(IValue.from(input)).toTensor();

        // getting tensor content as java array of floats
        final float[] scores = output.getDataAsFloatArray();

        // searching for the index with maximum score
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        return maxScoreIdx;
    }
}
