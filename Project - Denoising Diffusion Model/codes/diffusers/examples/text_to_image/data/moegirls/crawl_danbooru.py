"""
    Code Author: Andes Kei
    Usage: crawl_danbooru.py [-h] [--query QUERY] [--max_pages MAX_PAGES] [--output_dir OUTPUT_DIR] [--output_size OUTPUT_SIZE] [--nsfw NSFW]

    Scrape images and tags from Danbooru.

    Arguments:
    -h, --help            Show help message and exit
    --query QUERY         Search query enclosed by "", please check https://danbooru.donmai.us/wiki_pages/help:cheatsheet for more details
    --max_pages MAX_PAGES
                          Maximum number of pages
    --output_dir OUTPUT_DIR
                          Output directory
    --output_size OUTPUT_SIZE
                          Output size, set to 0 if do not need preprocessing
    --nsfw NSFW           Download nsfw image or not
"""
from argparse import ArgumentParser
from pathlib import Path
from urllib import parse
import cv2
import numpy as np
import os
import requests
import time

parser = ArgumentParser(description="Scrape images and tags from Danbooru.")
parser.add_argument("--query", type=str, default="", help='Search query enclosed by "", \
                     please check https://danbooru.donmai.us/wiki_pages/help:cheatsheet for more details')
parser.add_argument("--max_pages", type=int, default=1, help="Maximum number of pages")
parser.add_argument("--output_dir", type=Path, default="output", help="Output directory")
parser.add_argument("--output_size", type=int, default=0, help="Output size, set to 0 if do not need preprocessing")
parser.add_argument("--nsfw", type=str, default="False", help="Download nsfw image or not")

headers = {"User-Agent": "CrawlDanbooru/1.0"}

banned_tags = ["furry", "realistic", "3d", "magazine_scan", "scan", "pixel art", "screentones", "retro_artstyle", \
               "1940s (style)", "1950s (style)", "1960s (style)", "1970s (style)", "1980s (style)", "1990s (style)"]
unused_tags = ["translation request", "translated", "commentary", "commentary request", "commentary typo", \
               "character request", "bad id", "bad link", "bad pixiv_id", "bad twitter id", "bad tumblr id", \
               "bad deviantart id", "bad nicoseiga_id", "md5 mismatch", "cosplay request", "artist request", \
               "wide image", "author request"]
nsfw_tags = ["sex", "oral", "fellatio gesture", "tentacle sex", "nipples", "pussy", "vaginal", "pubic hair", \
             "anus", "ass focus", "penis", "cum", "condom", "sex toy"]

def resize(im, output_size):
    ratio = max(output_size / im.shape[0], output_size / im.shape[1])
    return cv2.resize(im, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

def get_img(args, post):
    if post["file_ext"] not in ["jpg", "png"]:
        return False
    tags = post["tag_string"].replace(" ", ", ").replace("_", " "  )
    tag_list = [tag for tag in tags.split(", ") if tag not in unused_tags]
    for banned_tag in banned_tags:
        if banned_tag in tag_list:
            return False
    if args.nsfw == "False":
        for nsfw_tag in nsfw_tags:
            if nsfw_tag in tag_list:
                return False
    img_name = str(post["id"]) + "." + post["file_ext"]
    img_file = args.output_dir / Path(img_name)
    if img_file.exists():
        return False
    img_url = post["file_url"] if "file_url" in post else None
    if (img_url is None):
        return False
    res = requests.get(img_url, headers=headers, stream=True)
    if args.output_size > 0:
        img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(args.output_dir) + "/" + img_name, resize(img, args.output_size))
    else:
        with img_file.open('wb') as file:
            for bytes in res.iter_content(chunk_size=128):
                file.write(bytes)  
    tags_file = args.output_dir / Path("metadata.jsonl")
    with tags_file.open('a') as file:
        file.write('{"file_name": "' + img_name + '", "text": "' + tags + '"}\n')
    return True

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    img_saved, error_count = 0, 0
    for i in range(1, args.max_pages + 1):
        try:
            print(f"loading page {i}")
            url = "https://danbooru.donmai.us/posts.json?tags=" + parse.quote_plus(args.query) + "&page=" + str(i)
            res = requests.get(url=url, headers=headers)
            if res.status_code == 200:
                for post in res.json():
                    if (get_img(args, post)):
                        img_saved += 1
            else:
                if res.status_code == 422:
                    print("you cannot search for more than 2 tags at a time")
                elif res.status_code == 500:
                    print("a server-side error occurred, possibly because the database timed out")
                elif res.status_code == 410:
                    print("page limit exceeded")
                break
        except Exception as e:
            print(e)
            i -= 1
            error_count += 1
            time.sleep(60)
            if (error_count > 10):
                break
    print(f"Scraped {img_saved} files")

if __name__ == "__main__":
    main(parser.parse_args())