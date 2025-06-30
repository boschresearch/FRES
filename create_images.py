""" Utility classes and functions related to FRES (ACL 2025).
Copyright (c) 2025 Robert Bosch GmbH
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
butWITHOUT ANYWARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from html2image import Html2Image
from PIL import Image
from tqdm import tqdm
import pandas as pd
import argparse
import os
import json
import random


def style_zebra(df, tid, dataset):
    df = df.style.set_table_styles([
        {'selector': 'th', 'props': [("text-align", "center")]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [(
            'background-color', '#f2f2f2'), ("text-align", "center"), ('padding', '6px')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [(
            'background-color', '#ffffff'), ("text-align", "center"), ('padding', '6px')]}
    ]).hide()
    styled_df = df.to_html(index=False)
    text_file = open(f"{dataset}/html/{tid}.html", "w", encoding="utf-8")
    text_file.write(styled_df)
    text_file.close()
    return


def style_zebra_h(df, tid, dataset):
    df = df.style.set_table_styles([
        {'selector': 'th:not(.index_name)', 'props': [(
            'background-color', '#ffe5cc'), ('color', 'black'), ('text-align', 'center'), ('padding', '8px')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [(
            'background-color', '#f2f2f2'), ("text-align", "center"), ('padding', '8px')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [(
            'background-color', '#ffffff'), ("text-align", "center"), ('padding', '8px')]}
    ]).format(precision=2)
    styled_df = df.to_html(index=False)
    text_file = open(f"{dataset}/html/{tid}.html", "w", encoding="utf-8")
    text_file.write(styled_df)
    text_file.close()
    return


def style_hover(df, tid, dataset):
    df = df.style.set_table_styles(
        [
            {'selector': '', 'props': [('border-collapse', 'collapse')]},
            {'selector': 'th', 'props': [('background-color', '#ffff99')]},
            {'selector': 'td', 'props': [('background-color', '#ffffcc'), ('border',
                                                                           '1px solid black'), ('text-align', 'center'), ('padding', '6px')]}
        ]).hide()

    styled_df = df.to_html(index=False)
    text_file = open(f"{dataset}/html/{tid}.html", "w", encoding="utf-8")
    text_file.write(styled_df)
    text_file.close()
    return


def style_hover_h(df, tid, dataset):

    index_names = {
        'selector': '',
        'props': 'border-collapse: collapse;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #ffff99; color: balck; text-align: center;padding: 8px;border: 1px solid black;'
    }
    cells = {
        'selector': 'td',
        'props': 'background-color: #ffffcc; color: black; text-align: center;padding: 8px;border: 1px solid black;'
    }
    df = df.style.format(precision=2).set_table_styles(
        [index_names, headers, cells])
    styled_df = df.to_html()
    text_file = open(f"{dataset}/html/{tid}.html", "w", encoding="utf-8")
    text_file.write(styled_df)
    text_file.close()
    return


def style_bordered(df, tid, dataset):
    df = df.style.set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
        {'selector': 'th, td', 'props': [
            ('border', '1px solid black'), ('padding', '6px'), ('text-align', 'center')]}
    ]).hide()
    styled_df = df.to_html(index=False)
    text_file = open(f"{dataset}/html/{tid}.html", "w", encoding="utf-8")
    text_file.write(styled_df)
    text_file.close()
    return


def style_bordered_h(df, tid, dataset):

    index_names = {
        'selector': '.index_name',
        'props': 'font-style: italic; color: darkgrey; font-weight:normal; text-align: center; padding: 8px;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'border: 1px solid black; text-align: center;padding: 8px;'
    }
    cells = {
        'selector': 'td',
        'props': 'border: 1px solid black; text-align: center;padding: 8px;'
    }
    df = df.style.format(precision=2).set_table_styles(
        [index_names, headers, cells])
    styled_df = df.to_html(index=False)
    text_file = open(f"{dataset}/html/{tid}.html", "w", encoding="utf-8")
    text_file.write(styled_df)
    text_file.close()
    return


def style_dark(df, tid, dataset):
    df = df.style.set_table_styles([
        {'selector': '', 'props': [
            ('border-collapse', 'collapse'), ('width', '50%')]},
        {'selector': 'th', 'props': [
            ('background-color', '#333'), ('color', 'white'), ('padding', '6px'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [
            ('background-color', '#555'), ('color', 'white'), ('padding', '6px'), ('text-align', 'center')]},
        {'selector': 'tbody tr:nth-child(odd) td',
         'props': [('background-color', '#444')]},
        {'selector': 'tbody tr:nth-child(even) td',
         'props': [('background-color', '#555')]}
    ]).hide()

    styled_df = df.to_html(index=False)
    text_file = open(f"{dataset}/html/{tid}.html", "w", encoding="utf-8")
    text_file.write(styled_df)
    text_file.close()
    return


def style_dark_h(df, tid, dataset):
    df = df.style.set_table_styles([
        {'selector': '', 'props': [('border-collapse', 'collapse')]},
        {'selector': 'th:not(.index_name)', 'props': [(
            'background-color', '#333'), ('color', 'white'), ('text-align', 'center'), ('padding', '8px')]},
        {'selector': 'td', 'props': [
            ('background-color', '#555'), ('color', 'white'), ('padding', '8px'), ('text-align', 'center')]},
        {'selector': 'tbody tr:nth-child(odd) td',
         'props': [('background-color', '#444')]},
        {'selector': 'tbody tr:nth-child(even) td',
         'props': [('background-color', '#555')]}
    ]).format(precision=2)
    styled_df = df.to_html(index=False)
    text_file = open(f"{dataset}/html/{tid}.html", "w", encoding="utf-8")
    text_file.write(styled_df)
    text_file.close()
    return


def crop_image(filename, new_name):
    img = Image.open(filename)
    pixels = img.load()
    target = [(x, y) for y in range(0, img.size[1]) for x in range(
        0, img.size[0]) if pixels[x, y] != (255, 255, 255) and x < 5000 and y < 8000]
    right = max([item[0] for item in target])
    bottom = max([item[1] for item in target])
    img = img.crop((0, 0, right+10, bottom+10))
    img.save(new_name)
    return


def generate_image(table_text, output_dir, table_id, dataset, style_id):
    table_df = pd.DataFrame(table_text[1:], columns=table_text[0])
    html_path = f"{output_dir}/html/{table_id}.html"
    # to html
    style_dict = {"1": style_bordered, "2": style_zebra,
                  "3": style_hover, "4": style_dark}
    style_dict[style_id](table_df, tid=table_id, dataset=dataset)
    # to image
    hti = Html2Image(browser='edge', custom_flags=[
        '--default-background-color=FFFFFF', '--hide-scrollbars', '--force-device-scale-factor=1'])
    hti.screenshot(html_file=html_path,
                   css_str='body {zoom: 4;}', save_as=f"{dataset}_{table_id}.jpg", size=(3000, 4000))
    crop_image(f"{dataset}_{table_id}.jpg", f"{dataset}_{table_id}.jpg")


def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.ourtput_dir)
        os.makedirs(f"{args.ourtput_dir}/html")
    with open(args.dataset_path, "r") as f:
        crt = [json.load(item) for item in f]
    for i, item in enumerate(crt):
        generate_image(item["table_text"], args.output_dir,
                       table_id=i, dataset="crt", style_id=random.randint(1, 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default="controlled_data.json")
    parser.add_argument('--output_dir', type=str,
                        default="images")
    args = parser.parse_args()
    main(args)
