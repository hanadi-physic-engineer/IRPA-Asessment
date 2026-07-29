import cv2
import time
import math
from ultralytics import YOLO

# ==========================================
# KONFIGURASI MODEL
# ==========================================
# Ubah dengan lokasi file best.pt yang sudah Anda download dari Google Drive/Colab
MODEL_PATH = "/home/an/irpa-asesment/2-keter-jarak-tabung-hidung/train_nose_tube_seg_baseline/weights/best.pt" 
CONFIDENCE_THRESHOLD = 0.5  # Hanya tampilkan deteksi dengan akurasi di atas 50%

def main():
    print(f"Mencoba memuat model dari: {MODEL_PATH}")
    
    try:
        model = YOLO(MODEL_PATH)
        print("Model berhasil dimuat!")
    except Exception as e:
        print(f"Error memuat model: {e}")
        print("Pastikan file model sudah didownload ke laptop Anda dan ditaruh di folder/path yang tepat.")
        return

    # Buka Webcam (0 adalah webcam bawaan laptop, bisa dicoba 1, 2, dst jika pakai eksternal)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam. Pastikan tidak ada aplikasi lain yang sedang memakainya.")
        return

    # Atur resolusi streaming webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n=====================================")
    print("Memulai Realtime Inference...")
    print("Tekan tombol 'q' pada keyboard untuk keluar.")
    print("=====================================\n")
    
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari webcam.")
            break
            
        # Hitung FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # Jalankan inference
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        result = results[0]
        
        # results[0].plot() akan otomatis menggambar kotak/poligon dan nama kelasnya
        annotated_frame = result.plot()

        # Ekstrak data untuk perhitungan jarak hidung ke mulut tabung
        class_names = result.names
        noses = []
        tubes = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for box, cls_id in zip(boxes, classes):
                name = class_names[int(cls_id)].lower()
                x1, y1, x2, y2 = box
                
                if "nose" in name:
                    # Titik tengah hidung
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    noses.append((cx, cy))
                elif "tube" in name or "pipette" in name or "pipet" in name:
                    # Asumsi: Mulut tabung adalah bagian tengah tepi atas (y1)
                    mouth_x = int((x1 + x2) / 2)
                    mouth_y = int(y1)
                    # Perkiraan panjang tabung dalam piksel (panjang diagonal/sisi terpanjang)
                    tube_length_px = max(x2 - x1, y2 - y1)
                    tubes.append((mouth_x, mouth_y, tube_length_px))

        # Logika Peringatan
        warning_triggered = False
        
        for tx, ty, tube_length_px in tubes:
            # Gambar titik biru di posisi mulut tabung
            cv2.circle(annotated_frame, (tx, ty), 6, (255, 0, 0), -1) 
            
            # Hitung rasio piksel ke cm (Acuan: panjang tabung = 15 cm)
            cm_per_pixel = 15.0 / tube_length_px if tube_length_px > 0 else 0
            
            for nx, ny in noses:
                # Gambar titik tengah hidung
                cv2.circle(annotated_frame, (nx, ny), 6, (0, 165, 255), -1) 
                
                # Hitung jarak piksel
                dist_px = math.hypot(tx - nx, ty - ny)
                
                # Konversi jarak ke centimeter
                dist_cm = dist_px * cm_per_pixel
                
                # Gambar garis penghubung kuning
                cv2.line(annotated_frame, (tx, ty), (nx, ny), (0, 255, 255), 2)
                
                # Tuliskan teks jarak dalam satuan cm di tengah-tengah garis
                mid_x = int((tx + nx) / 2)
                mid_y = int((ty + ny) / 2)
                cv2.putText(annotated_frame, f"{dist_cm:.1f} cm", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Jika jaraknya kurang dari 10 cm
                if dist_cm < 10.0:
                    warning_triggered = True

        # Tampilkan teks peringatan jika terlalu dekat
        if warning_triggered:
            cv2.putText(
                annotated_frame,
                "BAHAYA: MULUT TABUNG DEKAT HIDUNG!",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255), # Warna Merah
                3
            )

        # Tampilkan FPS di pojok kiri atas
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0), # Warna Hijau
            2
        )

        # Tampilkan Window Preview
        cv2.imshow("YOLO Realtime Stream", annotated_frame)

        # Tunggu input tombol keyboard selama 1ms
        # Keluar jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan resource (Matikan webcam dan tutup jendela)
    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai.")

if __name__ == "__main__":
    main()
