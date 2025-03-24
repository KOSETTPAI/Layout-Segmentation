import os
from pdf2image import convert_from_path

def split_pdf_to_images(input_pdf_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF pages to images
    images = convert_from_path(input_pdf_path)
    for page_number, image in enumerate(images, start=1):
        output_image_path = os.path.join(output_folder, f'page_{page_number}.png')
        image.save(output_image_path, 'PNG')
        print(f'Saved: {output_image_path}')

if __name__ == "__main__":
    input_pdf_path = r'newspaper\newspaper\07\24\km\24 July 24.pdf'  # Replace with your PDF file
    output_folder = r'newspaper_images\07\24\KM'  # Replace with your desired output folder
    split_pdf_to_images(input_pdf_path, output_folder)