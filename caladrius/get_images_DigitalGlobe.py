from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import urllib.request
import sys
import time
import os

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def main():
    output_dir = '../data/digital-globe/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(output_dir+'pre-event/')
        os.mkdir(output_dir + 'post-event/')

    # initialize webdriver
    opts = Options()
    opts.headless = True
    assert opts.headless  # operating in headless mode

    binary = r'C:\Program Files\Mozilla Firefox\firefox.exe'
    options = Options()
    options.set_headless(headless=True)
    options.binary = binary
    cap = DesiredCapabilities().FIREFOX
    cap["marionette"] = True  # optional
    browser = Firefox(firefox_options=options, capabilities=cap,
                 executable_path="C:\\geckodriver\\geckodriver.exe")
    print("Headless Firefox Initialized")
    base_url = 'view-source:https://www.digitalglobe.com/ecosystem/open-data/typhoon-mangkhut'
    browser.get(base_url)

    # find all images
    image_elements = browser.find_elements_by_css_selector('a')
    image_urls = [el.get_attribute('text') for el in image_elements]
    for url in image_urls:
        name = url.split('/')[-1]
        if not name.endswith('.tif'):
            continue
        try:
            if 'pre-event' in url:
                urllib.request.urlretrieve(url, output_dir+'pre-event/'+name, reporthook)
                break
            elif 'post-event' in url:
                urllib.request.urlretrieve(url, output_dir+'post-event/'+name, reporthook)
        except:
            continue

if __name__ == "__main__":
    main()