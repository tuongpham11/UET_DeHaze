# DISTS: Deep Image Structure and Texture Similarity

## Giới thiệu
DISTS (Deep Image Structure and Texture Similarity) là một metric đánh giá chất lượng hình ảnh hiện đại dựa trên deep learning. Metric này được thiết kế để đồng thời đánh giá cả cấu trúc (structure) và kết cấu (texture) của hình ảnh - hai yếu tố quan trọng trong nhận thức thị giác của con người.

**Reference:** Keyan Ding, Kede Ma, Shiqi Wang, and Eero P. Simoncelli. "Image Quality Assessment: Unifying Structure and Texture Similarity." IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.

## Ưu điểm của DISTS

1. **Perceptually aligned:** DISTS phù hợp với nhận thức thị giác con người tốt hơn so với các metrics truyền thống như PSNR và SSIM.

2. **Cân bằng structure và texture:** Nhiều metrics chỉ tập trung vào cấu trúc (như SSIM) hoặc chỉ tập trung vào sai số pixelwise (như PSNR). DISTS đánh giá cả hai.

3. **Dựa trên deep features:** Sử dụng features từ mạng VGG đã pre-trained, nắm bắt được các đặc trưng ngữ nghĩa của hình ảnh.

4. **Robust với nhiều loại biến dạng:** Hoạt động tốt với nhiều loại biến dạng khác nhau (blur, noise, compression, etc.).

## Cách hoạt động

DISTS kết hợp hai thành phần chính:

1. **Structure similarity:** Đo lường mức độ tương đồng về cấu trúc thông qua cross-correlation giữa các feature maps ở các tầng của VGG.

2. **Texture similarity:** Đo lường mức độ tương đồng về kết cấu thông qua việc so sánh các thống kê (mean, variance) của feature maps.

DISTS được tính bằng cách:
- Extract features từ VGG ở nhiều tầng (từ low-level đến high-level)
- Tính structure và texture similarity ở mỗi tầng
- Kết hợp các scores với các trọng số học được

## Thang điểm DISTS

- **Range:** 0 đến 1
- **Direction:** Càng thấp càng tốt (0 = giống hệt nhau, 1 = hoàn toàn khác nhau)
- **Threshold:** Thường < 0.2 được coi là chất lượng tốt

## So sánh với PSNR và SSIM

| Metric | Range | Ưu điểm | Nhược điểm |
|--------|-------|---------|------------|
| PSNR   | 0-inf (dB) | Đơn giản, nhanh | Không phản ánh nhận thức thị giác |
| SSIM   | 0-1 | Tốt cho structure | Không đánh giá đúng texture |
| DISTS  | 0-1 | Cân bằng structure và texture | Tính toán phức tạp hơn |

## Sử dụng DISTS trong training

DISTS đặc biệt hữu ích cho các task:

1. **Denoising:** Đảm bảo mô hình không chỉ loại bỏ nhiễu mà còn giữ nguyên kết cấu (texture) tự nhiên.

2. **Super-resolution:** Đánh giá khả năng khôi phục chi tiết một cách tự nhiên, không quá smooth.

3. **Compression artifact removal:** Đánh giá chất lượng phục hồi artifacts mà không làm mất texture.

## DISTS trong KAIR

Trong project KAIR, DISTS được thêm vào như một metric bổ sung cùng với PSNR. Trong khi PSNR tập trung vào độ chính xác pixel-wise, DISTS giúp đánh giá chất lượng nhận thức (perceptual quality) của hình ảnh được khôi phục.

### Hướng dẫn sử dụng

Trong quá trình testing, cả PSNR và DISTS đều được tính toán và log:

```
1--> image_001.png | PSNR: 32.56dB | DISTS: 0.0342
2--> image_002.png | PSNR: 30.12dB | DISTS: 0.0453
...
```

Sau khi hoàn thành test set, average metrics được hiển thị:

```
<epoch: 10, iter: 50000, Average PSNR: 31.78dB, Average DISTS: 0.0387>
```

### Phụ thuộc

DISTS đòi hỏi PyTorch và torchvision để chạy (sử dụng VGG16 pretrained). Đảm bảo rằng các dependencies này đã được cài đặt:

```bash
pip install torch torchvision
```

## Lưu ý

1. **Computational cost:** DISTS tốn kém về mặt tính toán hơn PSNR và SSIM vì cần forward qua mạng VGG.

2. **GPU acceleration:** Nên chạy trên GPU để tăng tốc độ tính toán.

3. **Batch processing:** Có thể tính toán DISTS cho cả batch để tăng hiệu suất.

4. **Pre-trained weights:** Sử dụng pre-trained weights của VGG16 trên ImageNet.

## Kết luận

DISTS là một metric hiện đại giúp đánh giá chất lượng hình ảnh toàn diện hơn. Khi kết hợp DISTS với PSNR, bạn có thể có cái nhìn đầy đủ về hiệu suất của mô hình - không chỉ về độ chính xác pixel-wise mà còn về chất lượng nhận thức.
