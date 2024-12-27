import cv2
from ultralytics import YOLO
from windowcapture import WindowCapture
from tkinter import Tk, Checkbutton, IntVar, Button, Label, Frame
from tkinter import font
import csv
import os
import numpy as np
from CriarGráficoCSV import criar_grafico_avaliacao, criar_grafico_performance  # Importa as funções de gráficos


# Função para aplicar pré-processamentos
def aplicar_preprocessamento(img, remove_noise, gaussian_blur, mean_blur, median_blur, bilateral_filter, clahe_eq, hist_eq):
    preproc_type = []  # Lista para armazenar os tipos de pré-processamento aplicados
    # Aplica os filtros conforme o checkBox do tkInter
    if remove_noise:
        img = img = cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 9, 9)

        preproc_type.append("Remoção de Ruído")
    if gaussian_blur:
        img = cv2.GaussianBlur(img, (5, 5), 0)
        preproc_type.append("Suavização Gaussiana")
    if mean_blur:
        img = cv2.blur(img, (5, 5))
        preproc_type.append("Suavização por Média")
    if median_blur:
        img = cv2.medianBlur(img, 5)
        preproc_type.append("Suavização pela Mediana")
    if bilateral_filter:
        img = cv2.bilateralFilter(img, 9, 75, 75)
        preproc_type.append("Suavização com Filtro Bilateral")
    if clahe_eq:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(img.shape) == 2:
            img = clahe.apply(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        preproc_type.append("Equalização de Histograma CLAHE")
    if hist_eq:
        if len(img.shape) == 2:
            img = cv2.equalizeHist(img)
        else:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = cv2.equalizeHist(lab_planes[0])
            lab = cv2.merge(lab_planes)
            img = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        preproc_type.append("Equalização de Histograma")
    return img, preproc_type  # Retorna a imagem e o tipo de pré-processamento


# Função para carregar dados existentes do CSV
def carregar_csv_dados(filename):
    data = {}
    if os.path.exists(filename):
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Pular o cabeçalho
            for row in reader:
                preproc, classes_str, avg_confidence = row
                data[preproc] = {
                    "classes_str": classes_str,
                    "avg_confidence": float(avg_confidence)
                }
    return data

# Função para salvar os melhores resultados no CSV
def salvar_melhores_resultados(filename, overall_performance):
    existing_data = carregar_csv_dados(filename)

    for preproc, data in overall_performance.items():
        avg_confidence = np.mean(data["confidences"])
        # Atualiza somente se a nova acurácia média for maior que a existente
        if (preproc in existing_data and avg_confidence > existing_data[preproc]["avg_confidence"]) or preproc not in existing_data:
            classes_str = ", ".join([f"{cls}/{conf:.2f}" for cls, conf in data["classes"].items()])
            existing_data[preproc] = {
                "classes_str": classes_str,
                "avg_confidence": avg_confidence
            }

    # Escrever dados atualizados de volta no CSV
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["pre-processamento", "classe/acuracia", "acuracia media"])  # Cabeçalho
        for preproc, data in existing_data.items():
            writer.writerow([preproc, data["classes_str"], f"{data['avg_confidence']:.2f}"])


# Função de detecção principal com chamada para salvar resultados
def deteccao_de_execucao():
    wincap = WindowCapture(size=(845, 1080), origin=(0, 0))
    model = YOLO("runs/detect/train/weights/best.pt")
    overall_performance = {}

    while True:
        img = wincap.get_screenshot()
        img, preproc_type = aplicar_preprocessamento(
            img,
            var_remove_noise.get(),
            var_gaussian_blur.get(),
            var_mean_blur.get(),
            var_median_blur.get(),
            var_bilateral_filter.get(),
            var_clahe_eq.get(),
            var_hist_eq.get()
        )

        if seguir:
            results = model.track(img, persist=True)
        else:
            results = model(img)

        for preproc in preproc_type:
            if preproc not in overall_performance:
                overall_performance[preproc] = {"classes": {}, "confidences": []}
            for result in results:
                for box in result.boxes:
                    class_idx = int(box.cls.item())
                    class_name = model.names.get(class_idx, "Classe desconhecida")
                    confidence = box.conf.item()
                    if class_name not in overall_performance[preproc]["classes"] or confidence > overall_performance[preproc]["classes"][class_name]:
                        overall_performance[preproc]["classes"][class_name] = confidence
                    overall_performance[preproc]["confidences"].append(confidence)
            img = result.plot()

        cv2.imshow("Tela", img)
        cv2.moveWindow("Tela", 660, 0)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

    # Salvar os melhores resultados
    csv_filename = "performance_data.csv"
    salvar_melhores_resultados(csv_filename, overall_performance)

# Função para alterar o estado dos checkboxes
def varCheckBox(var_to_toggle, *vars_to_uncheck):
    # Marca o checkbox atual e desmarca os outros
    var_to_toggle.set(1)
    for var in vars_to_uncheck:
        var.set(0)

# Interface do usuário para seleção de pré-processamento
def criar_interface():
    root = Tk()
    root.title("Seleção de Pré-Processamento")
    root.geometry("450x550")
    root.configure(bg='#f8f9fa')

    global var_remove_noise, var_gaussian_blur, var_mean_blur, var_median_blur, var_bilateral_filter, var_clahe_eq, var_hist_eq
    var_remove_noise = IntVar()
    var_gaussian_blur = IntVar()
    var_mean_blur = IntVar()
    var_median_blur = IntVar()
    var_bilateral_filter = IntVar()
    var_clahe_eq = IntVar()
    var_hist_eq = IntVar()

    header_font = font.Font(family="Helvetica", size=14, weight="bold")
    my_font = font.Font(family="Helvetica", size=12)

    header_frame = Frame(root, bg='#007bff', bd=2, relief="solid")
    header_frame.pack(fill='x', pady=10)
    header_label = Label(header_frame, text="Opções de Pré-Processamento", font=header_font, fg='white', bg='#007bff')
    header_label.pack(pady=10)

    option_frame = Frame(root, bg='#ffffff', padx=10, pady=10, bd=1, relief="groove")
    option_frame.pack(pady=10)

    # Criação dos checkboxes com a função toggle
    Checkbutton(option_frame, text="Remoção de Ruído", variable=var_remove_noise, command=lambda: varCheckBox(var_remove_noise, var_gaussian_blur, var_mean_blur, var_median_blur, var_bilateral_filter, var_clahe_eq, var_hist_eq), bg='#ffffff', font=my_font).pack(anchor='w', pady=5)
    Checkbutton(option_frame, text="Suavização Gaussiana", variable=var_gaussian_blur, command=lambda: varCheckBox(var_gaussian_blur, var_remove_noise, var_mean_blur, var_median_blur, var_bilateral_filter, var_clahe_eq, var_hist_eq), bg='#ffffff', font=my_font).pack(anchor='w', pady=5)
    Checkbutton(option_frame, text="Suavização por Média", variable=var_mean_blur, command=lambda: varCheckBox(var_mean_blur, var_remove_noise, var_gaussian_blur, var_median_blur, var_bilateral_filter, var_clahe_eq, var_hist_eq), bg='#ffffff', font=my_font).pack(anchor='w', pady=5)
    Checkbutton(option_frame, text="Suavização pela Mediana", variable=var_median_blur, command=lambda: varCheckBox(var_median_blur, var_remove_noise, var_gaussian_blur, var_mean_blur, var_bilateral_filter, var_clahe_eq, var_hist_eq), bg='#ffffff', font=my_font).pack(anchor='w', pady=5)
    Checkbutton(option_frame, text="Suavização com Filtro Bilateral", variable=var_bilateral_filter, command=lambda: varCheckBox(var_bilateral_filter, var_remove_noise, var_gaussian_blur, var_mean_blur, var_median_blur, var_clahe_eq, var_hist_eq), bg='#ffffff', font=my_font).pack(anchor='w', pady=5)
    Checkbutton(option_frame, text="Equalização CLAHE", variable=var_clahe_eq, command=lambda: varCheckBox(var_clahe_eq, var_remove_noise, var_gaussian_blur, var_mean_blur, var_median_blur, var_bilateral_filter, var_hist_eq), bg='#ffffff', font=my_font).pack(anchor='w', pady=5)
    Checkbutton(option_frame, text="Equalização de Histograma", variable=var_hist_eq, command=lambda: varCheckBox(var_hist_eq, var_remove_noise, var_gaussian_blur, var_mean_blur, var_median_blur, var_bilateral_filter, var_clahe_eq), bg='#ffffff', font=my_font).pack(anchor='w', pady=5)

    Button(root, text="INICIAR DETECÇÃO", command=deteccao_de_execucao, bg='#28a745', fg='white', font=my_font, bd=1, relief="solid").pack(pady=20)
    Button(root, text="AVALIAÇÃO GERAL", command=criar_grafico_avaliacao, bg='#007bff', fg='white', font=my_font, bd=1, relief="solid").pack(pady=5)
    Button(root, text="GRÁFICO DINÂMICO", command=criar_grafico_performance, bg='#9370DB', fg='white', font=my_font, bd=1, relief="solid").pack(pady=5)

    root.mainloop()


# Variáveis de controle
offset_x = 0
offset_y = 0
seguir = True

# Iniciar a interface
criar_interface()