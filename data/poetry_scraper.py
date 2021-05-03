# outdated, see data_analysis.ipynb for info about data
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

PATH = "C:\Program Files (x86)\chromedriver.exe"
# main driver for page finding
driver = webdriver.Chrome(PATH)

schools = {
    149: "Augustan", 150: "Beat", 151: "Black Mountain", 
    152: "Confessional", 153: "Fugitive", 154: "Georgian", 
    155: "Harlem Renaissance", 156: "Imagist", 157: "Language Poetry", 
    158: "Middle English", 159: "Modern", 160: "NY School", 
    161: "NY School (2.Gen)", 162: "Objectivist", 
    163: "Renaissance", 164: "Romantic", 165: "Victorian", 
    304: "Black Arts Movement"
    }

poems = dict()
po_id = 1

# search poems by school/period
for school in schools:
    id = school
    driver.get(f"https://www.poetryfoundation.org/poems/browse#page=1&sort_by=recently_added&school-period={id}")
    
    # find number of search result pages
    pagination = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "filter-pagination-select")))
    pages = pagination.find_elements_by_tag_name("option")

    page_number = [page.text for page in pages]
    for i in page_number:
        # secondary driver for traversing one-page search results
        driver2 = webdriver.Chrome(PATH)
        driver2.get(f"https://www.poetryfoundation.org/poems/browse#page={i}&sort_by=recently_added&school-period={id}")
        page = WebDriverWait(driver2, 15).until(EC.presence_of_element_located((By.CLASS_NAME , "c-vList c-vList_bordered c-vList_bordered_thorough")))
        # from each result page, find the titles and authors for each poem
        titles = page.find_elements_by_tag_name("a")
        authors = page.find_elements_by_class_name("c-feature-sub")

        # testing (doesn't find all titles and no authors)
        print(len(titles), len(authors))
        # if the numbers don't add up (more titles than authors), stop the process
        if len(titles) != len(authors):
            break

        # save the title, author and url of each poem as a tuple and save in the global poetry dictionary
        poem_page = [(title.text, title.get_attribute(href), authors.text[4:]) for title in titles for author in authors]
        for el in poem_page:
            poems[po_id] = {"title": el[0], "author": el[2], "school": schools[school], "url": el[1]}
            po_id += 1

driver.quit()
print(len(poems))