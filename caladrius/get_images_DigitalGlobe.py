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
    duration = max(time.time() - start_time, 1)
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def main():
    output_dir = '/mnt/data/digital-globe/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(output_dir+'pre-event/')
        os.mkdir(output_dir + 'post-event/')

    # initialize webdriver
    opts = Options()
    opts.headless = True
    assert opts.headless  # operating in headless mode

    # binary = r'C:\Program Files\Mozilla Firefox\firefox.exe'
    options = Options()
    options.set_headless(headless=True)
    # options.binary = binary
    cap = DesiredCapabilities().FIREFOX
    cap["marionette"] = True  # optional
    browser = Firefox(firefox_options=options, capabilities=cap) #,executable_path="C:\\geckodriver\\geckodriver.exe")
    print("Headless Firefox Initialized")
    base_url = 'view-source:https://www.digitalglobe.com/ecosystem/open-data/typhoon-mangkhut'
    browser.get(base_url)

    pre_event = open("pre-event.txt", "w")
    post_event = open("post-event.txt", "w")

    # find all images
    image_elements = browser.find_elements_by_css_selector('a')
    image_urls = [el.get_attribute('text') for el in image_elements]
    for url in image_urls:
        if '301332' not in url:
            continue
        print(url)
        name = url.split('/')[-1]
        if not name.endswith('.tif'):
            continue
        cat = url.split('/')[-2]

        if 'pre-event' in url:
            # urllib.request.urlretrieve(url, output_dir+'pre-event/'+cat+'-'+name, reporthook)
            pre_event.write(url)
        elif 'post-event' in url:
            # urllib.request.urlretrieve(url, output_dir+'post-event/'+cat+'-'+name, reporthook)
            post_event.write(url)

    pre_event.close()
    post_event.close()

if __name__ == "__main__":
    main()