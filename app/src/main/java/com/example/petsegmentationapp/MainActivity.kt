package com.example.petsegmentationapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var btnSelectImage: Button
    private lateinit var btnInfer: Button
    private lateinit var ivOriginal: ImageView
    private lateinit var ivMask: ImageView

    private var selectedBitmap: Bitmap? = null
    private var pyTorchModule: Module? = null

    // Função para abrir a galeria e receber o resultado
    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            val inputStream = contentResolver.openInputStream(uri)
            selectedBitmap = BitmapFactory.decodeStream(inputStream)
            ivOriginal.setImageBitmap(selectedBitmap)
            btnInfer.isEnabled = true // Ativa o botão de inferência
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Ligar as variáveis aos botões da interface XML
        btnSelectImage = findViewById(R.id.btnSelectImage)
        btnInfer = findViewById(R.id.btnInfer)
        ivOriginal = findViewById(R.id.ivOriginal)
        ivMask = findViewById(R.id.ivMask)

        // 1. Carregar o modelo assim que a app abre
        try {
            val modelPath = assetFilePath("unet_pet_fp32.pt")
            pyTorchModule = Module.load(modelPath)
        } catch (e: Exception) {
            Toast.makeText(this, "Erro ao carregar o modelo", Toast.LENGTH_LONG).show()
        }

        // 2. Ação do botão "Selecionar Imagem"
        btnSelectImage.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        // 3. Ação do botão "Realizar Inferência"
        btnInfer.setOnClickListener {
            if (selectedBitmap != null && pyTorchModule != null) {
                runInference(selectedBitmap!!)
            } else {
                Toast.makeText(this, "Selecione uma imagem primeiro", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun runInference(bitmap: Bitmap) {
        // Redimensionar a imagem para o tamanho que o modelo espera (128x128)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true)

        // Como no treino apenas dividimos por 255 (scale=True), usamos médias 0 e std 1
        val noMeanRGB = floatArrayOf(0.0f, 0.0f, 0.0f)
        val noStdRGB = floatArrayOf(1.0f, 1.0f, 1.0f)

        // Converter o Bitmap para Tensor [1, 3, 128, 128]
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, noMeanRGB, noStdRGB)

        // Passar a imagem pelo modelo
        val outputTensor = pyTorchModule!!.forward(IValue.from(inputTensor)).toTensor()

        // Extrair o resultado (Array de Floats)
        val outputArray = outputTensor.dataAsFloatArray

        // Processar a máscara: Encontrar qual classe teve a maior pontuação em cada pixel
        val width = 128
        val height = 128
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val idx = y * width + x

                // O modelo devolve 3 canais (um para cada classe: fundo, animal, borda)
                val c0 = outputArray[0 * width * height + idx] // Classe 0: Fundo
                val c1 = outputArray[1 * width * height + idx] // Classe 1: Animal
                val c2 = outputArray[2 * width * height + idx] // Classe 2: Borda

                // Descobrir qual canal tem o maior valor (ArgMax)
                var maxVal = c0
                var maxIdx = 0
                if (c1 > maxVal) { maxVal = c1; maxIdx = 1 }
                if (c2 > maxVal) { maxVal = c2; maxIdx = 2 }

                // Atribuir uma cor baseada na classe vencedora
                pixels[idx] = when (maxIdx) {
                    0 -> Color.BLACK   // Fundo -> Preto
                    1 -> Color.GREEN   // Animal -> Verde
                    2 -> Color.RED     // Borda -> Vermelho
                    else -> Color.BLACK
                }
            }
        }

        // Criar um novo Bitmap com a máscara resultante e exibir no ImageView
        val maskBitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        ivMask.setImageBitmap(maskBitmap)
    }

    // Função auxiliar para copiar o modelo da pasta assets para o armazenamento interno
    // O PyTorch precisa de um caminho de ficheiro real para carregar o modelo
    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }
}