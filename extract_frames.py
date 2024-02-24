import cv2
import os


def extract_using_cpu():
    # Caminho para a pasta com os vídeos
    video_folder = 'C:/Users/nikol/git/python/PerfectRepTraining/videos'

    # Caminho para a pasta onde os frames serão salvos
    frame_folder = 'C:/Users/nikol/git/python/PerfectRepTraining/frames'

    # Lista os arquivos na pasta de vídeos
    videos = os.listdir(video_folder)

    # Loop sobre cada vídeo na pasta
    for video_name in videos:
        video_path = os.path.join(video_folder, video_name)
        # Abre o vídeo
        cap = cv2.VideoCapture(video_path)

        # Cria a pasta para os frames do vídeo atual
        frame_output_folder = os.path.join(frame_folder, os.path.splitext(video_name)[0])
        os.makedirs(frame_output_folder, exist_ok=True)

        # Contador para o número de frames
        frame_count = 0

        # Loop para extrair e salvar os frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f'{frame_count:04d}.png'
            frame_path = os.path.join(frame_output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            frame_count += 1

            print(f'Frame {frame_name} gerado do vídeo {video_name}')

        # Fecha o vídeo
        cap.release()

    print('Frames extraídos com sucesso.')


if __name__ == '__main__':
    extract_using_cpu()
