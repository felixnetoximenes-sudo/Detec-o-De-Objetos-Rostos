# Detecção de Rostos e Objetos em Tempo Real com YOLOv8 🚀

Este projeto implementa um sistema de visão computacional em tempo real que captura o feed de vídeo da webcam, processa cada frame utilizando a inteligência artificial **YOLOv8 (Ultralytics)** baseada em **PyTorch**, e exibe as detecções na tela instantaneamente.

O objetivo deste repositório é demonstrar a integração prática entre manipulação de vídeo com OpenCV e inferência de modelos de Deep Learning para o meu portfólio de Visão Computacional.

---

## 🛠️ Tecnologias Utilizadas

* **Python** (Linguagem base)
* **OpenCV** (Captura e manipulação do fluxo de vídeo)
* **PyTorch** (Framework de Deep Learning que sustenta a IA)
* **Ultralytics YOLOv8** (Modelo de detecção de objetos de alta performance)

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
Certifique-se de ter o Python instalado em sua máquina. Em seguida, clone este repositório ou baixe os arquivos.

### 2. Instalar as Dependências
Abra o seu terminal ou prompt de comando na pasta do projeto e instale as bibliotecas necessárias:

```bash
pip install ultralytics opencv-python torch
```

### 3. Executar o Script
Rode o script principal:

```bash
python main.py
```

*Ao rodar, o script baixará automaticamente o modelo padrão `yolov8n.pt` na primeira execução caso você não o tenha localmente.*
*Para encerrar a captura da câmera, basta pressionar a tecla **'q'** no teclado.*

---

## 💻 O Código Principal (`main.py`)

```python
import cv2
from ultralytics import YOLO

# 1. Carrega o modelo YOLO pré-treinado
model = YOLO("yolov8n.pt") 

# 2. Abre a conexão com a webcam (índice 0)
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Erro: Câmera não encontrada!")
else:
    print("Câmera iniciada com sucesso. Pressione 'q' para sair.")
    
    while True:
        sucesso, frame = video.read()
        
        if not successo:
            print("Fim de vídeo ou falha ao capturar o frame.")
            break
        
        # 3. Executa a detecção do YOLO no frame atual (stream=True otimiza a performance)
        resultados = model(frame, stream=True)
        
        # 4. Desenha as caixas de detecção (bounding boxes) no frame
        frame_processado = frame
        for r in resultados:
            frame_processado = r.plot()
        
        # 5. Exibe o resultado em uma janela física
        cv2.imshow("Deteccao YOLO em Tempo Real", frame_processado)
        
        # 6. Aguarda 1 milissegundo e verifica se a tecla 'q' foi pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos de hardware de forma segura
    video.release()
    cv2.destroyAllWindows()
```

---

## 📂 Próximos Passos (Roadmap)
- [ ] Criar e rotular um dataset próprio focado apenas em rostos específicos.
- [ ] Treinar o modelo usando PyTorch para gerar um arquivo `.pt` personalizado.
- [ ] Exportar o modelo final para o formato **ONNX** para otimizar a velocidade de inferência em CPU.
