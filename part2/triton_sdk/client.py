import tritonclient.http
from tritonclient.http import InferenceServerClient
from tritonclient.http import InferInput, InferRequestedOutput
import cv2
import numpy as np

# Inisialisasi client Triton
url = "localhost:8000"  # Sesuaikan dengan alamat server Triton Anda
client = InferenceServerClient(url=url)

# Membaca gambar menggunakan OpenCV (tanpa preprocessing)
image_path = "tes.jpg"
image = cv2.imread(image_path)

# Pastikan gambar ada
if image is None:
    raise ValueError("Image not found")

# Ubah gambar menjadi format tensor yang sesuai
# Gambar tetap dalam format (H, W, 3) untuk input model tanpa perubahan
image_tensor = np.ascontiguousarray(image)

# Menyiapkan input untuk Triton
inputs = []
inputs.append(InferInput('preprocessing_input', image_tensor.shape, "UINT8"))
inputs[0].set_data_from_numpy(image_tensor)

# Menyiapkan output untuk Triton
outputs = []
outputs.append(InferRequestedOutput('preprocessing_output'))

# Mengirim permintaan inferensi ke Triton
response = client.infer(model_name="preprocessing", inputs=inputs, outputs=outputs)

# Ambil hasil output
output_data = response.as_numpy('preprocessing_output')

# Proses output jika diperlukan
print("Output:", output_data.shape)
