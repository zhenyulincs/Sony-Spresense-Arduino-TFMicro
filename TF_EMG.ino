

#include "tensorflow/lite/micro/micro_interpreter.h"


#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "tensorflow/lite/schema/schema_generated.h"


#include "tensorflow/lite/micro/spresense/debug_log_callback.h"
#include "M5_best_float32.h"

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 512 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void debug_log_printf(const char* s) {
  Serial.print("Error: ");
  Serial.println(s);
}

void setup() {
  Serial.begin(115200);
  tflite::InitializeTarget();
  RegisterDebugLogCallback(debug_log_printf);
  memset(tensor_arena, 0, kTensorArenaSize * sizeof(uint8_t));


  model = tflite::GetModel(saved_model_M5_best_float32_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version "
                  + String(model->version()) + " not equal "
                  + "to supported version "
                  + String(TFLITE_SCHEMA_VERSION));
    return;
  } else {
    Serial.println("Model version: " + String(model->version()));
  }

  // This pulls in all the operation implementations we need.
  static tflite::MicroMutableOpResolver<5> resolver;

  resolver.AddAveragePool2D();
  resolver.AddConv2D();
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddReshape();

    // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  } else {
    Serial.println("AllocateTensor() Success");
  }

  size_t used_size = interpreter->arena_used_bytes();
  Serial.println("Area used bytes: " + String(used_size));
  input = interpreter->input(0);
  output = interpreter->output(0);

  /* check input */


  Serial.println("Model input:");
  Serial.println("input->type: " + String(input->type));
  Serial.println("dims->size: " + String(input->dims->size));
  for (int n = 0; n < input->dims->size; ++n) {
    Serial.println("dims->data[n]: " + String(input->dims->data[n]));
  }

  Serial.println("Model output:");
  Serial.println("dims->size: " + String(output->dims->size));
  for (int n = 0; n < output->dims->size; ++n) {
    Serial.println("dims->data[n]: " + String(output->dims->data[n]));
  }



  for (int i = 0; i < 192*20; ++i) {
    input->data.f[i] = (float)(0.0);
  }

  Serial.println("Do inference");
  uint32_t start_time = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }
  uint32_t duration = micros() - start_time;
  Serial.println("Inference time = " + String(duration));

  for (int n = 0; n < 8; ++n) {
    float value = output->data.f[n];
    Serial.println("[" + String(n) + "] " + String(value));
  }
}
void loop() {}