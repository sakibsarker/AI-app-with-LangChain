pages = loader.load_and_split()
print(len(pages))
page=pages[0]
# print(page)
print(page.page_content[0:700])
print(page.metadata)