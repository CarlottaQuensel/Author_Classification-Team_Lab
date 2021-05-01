from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)

schools = {149: "Augustan", 150: "Beat", 151: "Black Mountain", 
152: "Confessional", 153: "Fugitive", 154: "Georgian", 
155: "Harlem Renaissance", 156: "Imagist", 157: "Language Poetry", 
158: "Middle English", 159: "Modern", 160: "NY School", 
161: "NY School (2.Gen)", 162: "Objectivist", 
163: "Renaissance", 164: "Romantic", 165: "Victorian", 
304: "Black Arts Movement"}

for school in schools:
    id = school
    driver.get(f"https://www.poetryfoundation.org/poems/browse#page=1&sort_by=recently_added&school-period={id}")
    
    pagination = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "filter-pagination-select")))
    
    pages = pagination.find_elements_by_tag_name("option")
    page_link = [page.text for page in pages]
    #for i in page_link:
    #    driver.get(f"https://www.poetryfoundation.org/poems/browse#page={i}&sort_by=recently_added&school-period={id}")


print(page_link)
driver.quit()
