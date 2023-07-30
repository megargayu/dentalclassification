package com.ayush.dentalclassification;

import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import static com.ayush.dentalclassification.ModelManager.RandomImageResult;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
  ModelManager modelManager;

  private void updateStatuses() {
    Spinner modelDropdown = findViewById(R.id.model_dropdown);
    String modelStr = modelDropdown.getSelectedItem().toString();

    TextView modelStatus = findViewById(R.id.model_status);
    TextView datasetStatus = findViewById(R.id.dataset_status);

    Button randomImageBtn = findViewById(R.id.random_image_button);
    Button fullTestBtn = findViewById(R.id.full_test_button);

    // Check Model Status
    ModelManager.Model model = modelManager.models.get(modelStr);
    assert model != null;

    boolean modelLoaded = model.module != null;
    modelStatus.setText(modelLoaded ? "\uD83D\uDFE2 Model Loaded" : "\uD83D\uDD34 Model Not Loaded");

    // Check Dataset Status
    boolean datasetLoaded = modelManager.images.size() != 0;
    datasetStatus.setText(datasetLoaded ? "\uD83D\uDFE2 Dataset Loaded" : "\uD83D\uDD34 Dataset Not Loaded");

    // Update buttons
    if (modelLoaded && datasetLoaded) {
      randomImageBtn.setEnabled(true);
      fullTestBtn.setEnabled(true);
    } else {
      randomImageBtn.setEnabled(false);
      fullTestBtn.setEnabled(false);
    }
  }

  @SuppressLint("DefaultLocale")
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    try {
      modelManager = new ModelManager(this);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    ImageView imageView = findViewById(R.id.image);
    TextView textView = findViewById(R.id.text);

    Button loadDatasetButton = findViewById(R.id.load_dataset);

    loadDatasetButton.setOnClickListener(view -> {
      try {
        modelManager.loadAllImages();
        updateStatuses();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });

    Spinner modelDropdown = findViewById(R.id.model_dropdown);
    Button loadModelBtn = findViewById(R.id.load_model);

    modelDropdown.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
      @Override
      public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
        updateStatuses();
      }

      @Override
      public void onNothingSelected(AdapterView<?> adapterView) {
      }
    });

    loadModelBtn.setOnClickListener(view -> {
      String modelStr = modelDropdown.getSelectedItem().toString();

      try {
        ModelManager.Model model = modelManager.models.get(modelStr);
        assert model != null;

        model.module = modelManager.loadModule(model.path);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

      updateStatuses();
    });

    Button randomImageBtn = findViewById(R.id.random_image_button);
    randomImageBtn.setOnClickListener(view -> {
      System.out.println("random image button clicked!");
      try {
        RandomImageResult result = modelManager.runRandomImage(modelDropdown.getSelectedItem().toString());
        System.out.println("Ran model!");

        imageView.setImageBitmap(result.image);
        textView.setText(String.format("Output: %.1f | Truth: %.1f", result.output, result.groundTruth));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });

    Button fullTestBtn = findViewById(R.id.full_test_button);
    fullTestBtn.setOnClickListener(view -> {
      try {
        float accuracy = modelManager.runFullTest(modelDropdown.getSelectedItem().toString());

        textView.setText(String.format("Accuracy: %.5f", accuracy));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Copies specified asset to the file in /files app directory and returns this file absolute path.
   *
   * @return absolute file path
   */
  public static String assetFilePath(Context context, String assetName) throws IOException {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    System.out.println();

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }
}
