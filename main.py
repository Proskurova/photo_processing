import cv2
from pytesseract import pytesseract
from natasha import (
    Segmenter,
    MorphVocab,
    PER,
    NamesExtractor,
    NewsNERTagger,
    NewsEmbedding,
    Doc
)


def photo_transmission(photo):
    img = cv2.imread(f'image/{photo}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU |
                                              cv2.THRESH_BINARY_INV)
    cv2.imwrite('threshold_image.jpeg', thresh1)
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

    dilation = cv2.dilate(thresh1, rect_kernel, iterations=3)
    cv2.imwrite('dilation_image.jpeg', dilation)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    crop_number = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Рисуем ограничительную рамку на текстовой области
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Обрезаем область ограничительной рамки
        cropped = im2[y:y + h, x:x + w]

        cv2.imwrite("crop" + str(crop_number) + ".jpeg", cropped)
        crop_number += 1

        cv2.imwrite('rectanglebox.jpeg', rect)

        # Использование tesseract на обрезанной области изображения для получения текста
        text = pytesseract.image_to_string(cropped, lang='rus')

        # Добавляем текст в файл
        open_file_write(text)


def open_file_write(result):
    with open("text_output2.txt", "w", encoding="utf-8") as f:
        f.write(f"{result}\n")


def open_file_read():
    with open("text_output2.txt", encoding="utf-8") as f:
        return " ".join(f.read().split("\n")).title()


def finding_full_name():
    emb = NewsEmbedding()
    morph_vocab = MorphVocab()
    names_extractor = NamesExtractor(morph_vocab)
    text = open_file_read().replace('|', '').replace('=', '').replace('„', '').replace('.', '')
    print(text)
    doc = Doc(text)
    doc.segment(Segmenter())
    doc.tag_ner(NewsNERTagger(emb))

    for span in doc.spans:
        span.normalize(morph_vocab)
        if span.type == PER:
            span.extract_fact(names_extractor)
    name = [_.normal for _ in doc.spans if _.fact]
    print(name)


photo_transmission("9.jpeg")
finding_full_name()
