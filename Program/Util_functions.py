import codecs


def read_markdown_file(markdown_file):
    fileObj = codecs.open(markdown_file, 'r', 'utf_8_sig')
    text = fileObj.read()  # или читайте по строке
    fileObj.close()
    return text
