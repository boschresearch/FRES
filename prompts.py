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


PROMPT_NT = ("Please inspect the table(s) and then provide an answer to the question. Besides, your final answer should be in the JSON format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}. Directly return the final answer and nothing else.\nTable: {input_seg}\nQuestion:{question}",
             "Let's pretend you are an expert in reading table and answer questions. Return the answer in json format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}. Directly return the final answer and nothing else. \nTable: {input_seg}\nQuestion:{question}",
             "Please think step by step and return the answer in json format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}.\nTable: {input_seg}\nQuestion:{question}"
             )

PROMPT_NT_IMG = ("Please inspect the table(s) and then provide an answer to the question. Besides, your final answer should be in the JSON format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}. Directly return the final answer and nothing else.\nQuestion:{question}",
                 "Let's pretend you are an expert in reading table and answer questions. Return the answer in json format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}. Directly return the final answer and nothing else. \nQuestion:{question}",
                 "Please think step by step and return the answer in json format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}.\nQuestion:{question}"
                 )

PROMPT_NT_IMG_COLS = ("Please inspect the table(s) and then provide an answer to the question. Besides, your final answer should be in the JSON format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}. Directly return the final answer and nothing else. Pay close attention to the following columns in the table, as they contain relevant information for answering the question. Relevant columns: {col_names}\nQuestion:{question}",
                      "Let's pretend you are an expert in reading table and answer questions. Return the answer in json format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}. Directly return the final answer and nothing else.. Pay close attention to the following columns in the table, as they contain relevant information for answering the question. Relevant columns: {col_names}\nQuestion:{question}",
                      "Please think step by step and return the answer in json format {{\"answer\": [<a list of answer strings>]}} such as {{\"answer\": [\"87.56\", \"12.43\"]}}.Pay close attention to the following columns in the table, as they contain relevant information for answering the question. Relevant columns: {col_names}\nQuestion:{question}"
                      )
