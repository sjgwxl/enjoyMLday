import argparse
from PIL import Image
import os

def parse_args():
    parser = argparse.ArgumentParser(description='批量转换图片为灰度并调整分辨率')
    parser.add_argument('input_dir', type=str, help='输入图片所在目录')
    parser.add_argument('output_subdir', type=str, help='输出子目录名称（将在输入目录下创建）')
    parser.add_argument('--width', type=int, required=True, help='目标宽度（像素）')
    parser.add_argument('--height', type=int, required=True, help='目标高度（像素）')
    return parser.parse_args()


def main():
    # args = parse_args()
    # input_dir = args.input_dir
    # output_subdir = args.output_subdir
    # target_width = args.width
    # target_height = args.height
    input_dir = "D:\\ZK_WORK\\working\\PyCharmProject\\learningPython\\训练集"
    output_subdir = "D:\\ZK_WORK\\working\\PyCharmProject\\learningPython\\训练集\\out"
    target_width = 20
    target_height = 20

    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    output_dir = os.path.join(input_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    # 关键修改：仅遍历当前目录（非递归）
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)

        # 跳过子目录
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file)[1].lower()
        if ext not in supported_ext:
            skipped_count += 1
            continue

        try:
            with Image.open(file_path) as img:
                gray_img = img.convert('L')
                resized_img = gray_img.resize(
                    (target_width, target_height),
                    resample=Image.LANCZOS
                )
                output_path = os.path.join(output_dir, file)
                resized_img.save(output_path)
                processed_count += 1

        except Exception as e:
            print(f"处理失败: {file_path} - {str(e)}")
            skipped_count += 1

    print(f"✅ 成功处理：{processed_count} 个文件")
    print(f"⏭️ 跳过文件：{skipped_count} 个文件")
    print(f"📁 输出目录：{output_dir}")


if __name__ == '__main__':
    main()
