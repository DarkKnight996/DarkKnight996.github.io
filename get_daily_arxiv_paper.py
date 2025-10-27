#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的论文处理脚本
只支持单个日期，不再支持日期段
"""

import requests
import xml.etree.ElementTree as ET
import json
import csv
import os
import re
import tempfile
from datetime import datetime, timedelta
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm

# PDF处理相关
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("警告: PyPDF2未安装，无法处理PDF文件。请运行: pip install PyPDF2")

class CompletePaperProcessor:
    def __init__(self, docs_daily_path="docs/daily", temp_dir="temp_pdfs"):
        """
        初始化完整的论文处理器
        
        Args:
            docs_daily_path (str): daily文件夹路径
            temp_dir (str): 临时PDF存储目录
        """
        self.docs_daily_path = docs_daily_path
        self.temp_dir = temp_dir
        self.ensure_directories()
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
    
    def ensure_directories(self):
        """确保必要的目录存在"""
        for directory in [self.docs_daily_path, self.temp_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    # ==================== arXiv论文获取功能 ====================

    def fetch_arxiv_papers(self, categories=['cs.DC', 'cs.AI'], max_results=2000, target_date=None):
        """
        从arXiv获取指定分类的论文，并根据papers.jsonl去重与增补
        
        Args:
            categories (list): 论文分类列表
            max_results (int): 最大获取数量
            target_date (str): 目标日期，格式为 'YYYY-MM-DD'，本函数只考虑单个日期
            
        Returns:
            list: 论文列表（未在papers.jsonl中出现的新论文）
        """
        all_papers = []
        seen_papers = set()
        if not isinstance(target_date, str):
            print("只支持单个日期！target_date应为'YYYY-MM-DD'字符串。")
            return []

        # 收集arXiv数据
        for category in categories:
            print(f"正在获取 {category} 分类的论文信息...")
            search_query = f'cat:{category}'
            params = {
                'search_query': search_query,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            try:
                response = requests.get('http://export.arxiv.org/api/query', params=params, timeout=30)
                response.raise_for_status()
                root = ET.fromstring(response.content)
                ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
                print(f"Found {len(root.findall('arxiv:entry', ns))} papers")
                for entry in root.findall('arxiv:entry', ns):
                    paper_info = self._extract_paper_info(entry, ns)
                    if paper_info:
                        paper_id = paper_info.get('id', '')
                        if paper_id in seen_papers:
                            print(f"跳过重复论文: {paper_info.get('title', 'N/A')}")
                            continue
                        should_add = False
                        if category == 'cs.AI' or category == 'cs.LG':
                            summary_lower = paper_info.get("summary", "").lower()
                            if (
                                "accelerate" in summary_lower
                                or "accelerating" in summary_lower
                                or "acceleration" in summary_lower
                            ):
                                should_add = True
                        else:
                            should_add = True
                        if should_add:
                            all_papers.append(paper_info)
                            seen_papers.add(paper_id)
                print(f"成功获取 {category} 分类 {len([p for p in all_papers if any(cat in p.get('categories', []) for cat in [category])])} 篇论文")
                for i, paper in enumerate([p for p in all_papers if any(cat in p.get('categories', []) for cat in [category])]):
                    print(f"{i+1}. {paper.get('title', 'N/A')}")
            except Exception as e:
                print(f"获取 {category} 分类论文失败: {e}")

        print(f"去重后总共 {len(all_papers)} 篇论文")

        # 读取papers.jsonl已有的论文id
        papers_jsonl_path = os.path.join(self.docs_daily_path, "papers.jsonl")
        existing_ids = set()
        if os.path.exists(papers_jsonl_path):
            with open(papers_jsonl_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        paper = json.loads(line.strip())
                        pid = paper.get("id")
                        if pid:
                            existing_ids.add(pid)
                    except Exception:
                        continue

        # 过滤掉已在papers.jsonl中的论文
        filtered_papers = [p for p in all_papers if p.get("id") not in existing_ids]

        print(f"过滤后剩余 {len(filtered_papers)} 篇论文需要添加到papers.jsonl")

        # 将新论文追加进papers.jsonl，并准备返回列表
        fout = None
        with open(papers_jsonl_path, "a", encoding="utf-8") as fout:
            fout.write("\n")
            for paper in filtered_papers:
                out_item = {
                    "announced_date": target_date,
                    "categories": paper.get("categories", []),
                    "title": paper.get("title", ""),
                    "id": paper.get("id", ""),
                    "published_date": paper.get("published", ""),
                    "updated_date": paper.get("updated", "")
                }
                fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")

        for i, paper in enumerate(filtered_papers):
            print(f"{i+1}. {paper.get('title', 'N/A')}")

        print(f"total {len(filtered_papers)} new papers added to papers.jsonl")

        return filtered_papers
    
    def _extract_paper_info(self, entry, ns):
        """从XML条目中提取论文信息"""
        try:
            # 提取基本信息
            title_elem = entry.find('arxiv:title', ns)
            title = title_elem.text.strip() if title_elem is not None else "N/A"
            
            # 提取作者信息
            authors = []
            for author in entry.findall('arxiv:author', ns):
                name_elem = author.find('arxiv:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # 提取摘要
            summary_elem = entry.find('arxiv:summary', ns)
            summary = summary_elem.text.strip() if summary_elem is not None else "N/A"
            
            # 提取时间信息
            published_elem = entry.find('arxiv:published', ns)
            published = published_elem.text.strip() if published_elem is not None else "N/A"
            
            updated_elem = entry.find('arxiv:updated', ns)
            updated = updated_elem.text.strip() if updated_elem is not None else "N/A"
            
            # 提取链接
            pdf_link = "N/A"
            for link in entry.findall('arxiv:link', ns):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href', "N/A")
                    break
            
            # 提取arXiv ID
            arxiv_id = entry.find('arxiv:id', ns)
            paper_id = arxiv_id.text.strip() if arxiv_id is not None else "N/A"
            
            # 提取分类
            categories = []
            for category in entry.findall('arxiv:category', ns):
                if category.get('term'):
                    categories.append(category.get('term'))
            
            return {
                'id': paper_id,
                'title': title,
                'authors': authors,
                'summary': summary,
                'published': published,
                'updated': updated,
                'pdf_link': pdf_link,
                'categories': categories,
                'author_count': len(authors)
            }
            
        except Exception as e:
            print(f"提取论文信息时发生错误: {e}")
            return None

    def filter_by_updated_date(self, papers, date_str):
        """根据updated日期筛选论文"""
        filtered_papers = []
        for paper in papers:
            updated_field = paper.get('updated', '')
            try:
                dt = datetime.fromisoformat(updated_field.replace('Z', ''))
                if dt.strftime('%Y-%m-%d') == date_str:
                    filtered_papers.append(paper)
            except Exception:
                pass
        return filtered_papers

    # 日期段相关功能移除，不再支持
    # def filter_by_updated_date_range(self, papers, start_date, end_date):
    #     ...

    # ==================== PDF处理和LLM分析功能 ====================
    # ...无更改，省略...

    def download_pdf(self, pdf_url, filename):
        """下载PDF文件"""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return filepath
        except Exception as e:
            print(f"下载PDF失败 {pdf_url}: {e}")
            return None

    def extract_first_page_text(self, pdf_path):
        """提取PDF第一页的文本内容"""
        if not PDF_AVAILABLE:
            return "PDF处理库未安装"
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if len(pdf_reader.pages) > 0:
                    first_page = pdf_reader.pages[0]
                    text = first_page.extract_text()
                    return text[:4096]  # 限制长度避免API调用过长
                else:
                    return "PDF文件为空"
        except Exception as e:
            print(f"提取PDF文本失败 {pdf_path}: {e}")
            return f"PDF处理错误: {e}"

    def call_api_for_tags_institution_interest(self, title, abstract, first_page_text):
        # ...实现保持不变...
        prompt = f"""\
Title: {title}
Abstract: {abstract}
First Page Content: {first_page_text}

请为以上文章分类标签，一共有三层标签，分别为tag1, tag2, tag3。首先根据是否是sys分类为ai, sys/mlsys, 接着继续分类sys/mlsys，根据和LLM或者diffusion或者machine learning或者deep learning或者AI有关，只要有关就是mlsys，否则就是sys，第一个标签tag1为mlsys/sys，第二层标签tag2更细粒度，比如如果是mlsys，那就细分为: ai for science, LLM inference, LLM training, post-training, Other models training, Other models inference, edge computing, checkpointing, finetuning, trace analysis, cluster infrastructure, scheduling, kernels, security, federated learning, others这几种，如果是sys就分为hardware, compiler, quantum computing, operating system, cluster management, memory, network, filesystem, computation, fault-tolerance, security, programming languages, serverless, others这几种。第三层tag就根据文章内容总结关键词进行分类，第三层的标签可以是list，用逗号隔开。

另外，请根据作者信息和第一页内容推断论文的主要研究机构，可能会有多个机构，如果没有机构名的话，从作者的邮箱后缀判断。

最后，帮我判断我是否会对这篇文章感兴趣，判断标准如下：
- 如果tag1是mlsys，且tag2不是security, edge computing, mobile computing, federated learning, ai for science时，我都感兴趣；

请以如下格式输出，并在最后输出2-3句话对论文的主要方法和结论进行简单LLM总结（英文即可），不带多余解释说明或代码块：

tag1: <tag1>
tag2: <tag2>
tag3: <tag3, tag3, ...>
institution: <institution>
is_interested: <yes/no>
llm_summary: <2-3 sentences simple summary (method+conclusion)>
"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. You are good at summarizing papers and extracting keywords and institutions."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            result = response.choices[0].message.content.strip()
            
            # 解析结果
            lines = [line.strip() for line in result.splitlines() if line.strip()]
            tag1, tag2, tag3, institution, is_interested, llm_summary = "", "", "", "", "no", ""
            reading_summary = False
            summary_lines = []
            
            for line in lines:
                if line.lower().startswith("tag1:"):
                    tag1 = line.split(":", 1)[1].strip()
                elif line.lower().startswith("tag2:"):
                    tag2 = line.split(":", 1)[1].strip()
                elif line.lower().startswith("tag3:"):
                    tag3 = line.split(":", 1)[1].strip()
                elif line.lower().startswith("institution:"):
                    institution = line.split(":", 1)[1].strip()
                elif line.lower().startswith("is_interested:"):
                    is_interested = line.split(":", 1)[1].strip().lower()
                elif line.lower().startswith("llm_summary:"):
                    reading_summary = True
                    summary_line = line.split(":", 1)[1].strip()
                    if summary_line:
                        summary_lines.append(summary_line)
                elif reading_summary:
                    summary_lines.append(line)
            
            if summary_lines:
                llm_summary = ' '.join(summary_lines).strip()
            
            tag3_list = [t.strip() for t in tag3.split(',') if t.strip()]
            is_interested_bool = is_interested == "yes"
            return tag1, tag2, tag3_list, institution, is_interested_bool, llm_summary

        except Exception as e:
            print(f"API调用失败: {e}")
            return "", "", [], "", False, ""

    def process_single_paper(self, paper):
        # ...实现不变...
        title = paper.get('title', '')
        summary = paper.get('summary', '')
        pdf_link = paper.get('pdf_link', '')
        
        print(f"处理论文: {title}")
        
        # 下载PDF
        if not pdf_link or pdf_link == 'N/A':
            print(f"跳过论文 {title}: 无PDF链接")
            paper['is_interested'] = False
            return paper
        
        # 生成PDF文件名
        pdf_filename = f"{paper.get('id', '').split('/')[-1]}.pdf"
        
        # 下载PDF
        pdf_path = self.download_pdf(pdf_link, pdf_filename)
        if not pdf_path:
            print(f"跳过论文 {title}: PDF下载失败")
            paper['is_interested'] = False
            return paper
        
        # 提取第一页文本
        first_page_text = self.extract_first_page_text(pdf_path)
        
        # 调用API获取标签、机构和兴趣，并获取LLM总结
        tag1, tag2, tag3_list, institution, is_interested, llm_summary = self.call_api_for_tags_institution_interest(
            title, summary, first_page_text
        )
        
        # 更新论文信息
        paper['tag1'] = tag1
        paper['tag2'] = tag2
        paper['tag3'] = ', '.join(tag3_list)
        paper['institution'] = institution
        paper['is_interested'] = is_interested
        paper['llm_summary'] = llm_summary
        
        # 清理临时PDF文件
        try:
            os.remove(pdf_path)
        except:
            pass
        
        print(f"完成论文 {title}: tag1={tag1}, tag2={tag2}, institution={institution}, is_interested={'yes' if is_interested else 'no'}")
        return paper
    
    # ==================== Markdown文件处理功能 ====================
    # ...实现不变，省略...
    def get_week_range(self, date_str):
        """根据日期获取该周的周一到周日的日期范围"""
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
            days_since_monday = target_date.weekday()
            monday = target_date - timedelta(days=days_since_monday)
            sunday = monday + timedelta(days=6)
            
            start_str = monday.strftime('%Y%m%d')
            end_str = sunday.strftime('%Y%m%d')
            
            return f"{start_str}-{end_str}"
        except ValueError as e:
            print(f"日期格式错误: {e}")
            return None
    
    def get_arxiv_prefix(self, date_str):
        """根据日期获取类似[arXiv2510]的字符串"""
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            prefix = f"[arXiv{str(dt.year)[-2:]}{dt.month:02d}]"
            return prefix
        except Exception:
            return ""

    def format_paper_with_enhanced_info(self, paper, date_str=None):
        # ...实现不变...
        title = paper.get('title', 'N/A')
        authors = ', '.join(paper.get('authors', []))
        pdf_link = paper.get('pdf_link', 'N/A')
        
        # 使用API获取的标签和机构信息
        tags = []
        if paper.get('tag1'):
            tags.append("[" + paper['tag1'] + "]")
        if paper.get('tag2'):
            tags.append("[" + paper['tag2'] + "]")
        if paper.get('tag3'):
            tag3_items = [t.strip() for t in paper['tag3'].split(',') if t.strip()]
            if tag3_items:
                tags.append('[' + ', '.join(tag3_items) + ']')
        tags_str = ', '.join(tags) if tags else 'TBD'
        institution = paper.get('institution', 'TBD')
        llm_summary = paper.get('llm_summary', '').strip()
        
        # 获取arXiv前缀
        arxiv_prefix = ""
        if date_str is not None:
            arxiv_prefix = self.get_arxiv_prefix(date_str)
        else:
            # 兼容旧代码，如果没有传则从id猜测日期
            paper_id = paper.get("id", "")
            match = re.search(r'(\d{4,})(\d{2})', paper_id)
            if match:
                year = match.group(1)
                month = match.group(2)
                if year and month:
                    arxiv_prefix = f"[arXiv{year[-2:]}{month}]"
        
        formatted_text = f"""- **{arxiv_prefix} {title}**
  - **tags:** {tags_str}
  - **authors:** {authors}
  - **institution:** {institution}
  - **link:** {pdf_link}
"""
        if llm_summary:
            # 转义HTML特殊字符，避免MDX解析错误
            escaped_summary = llm_summary.replace('<', '&lt;').replace('>', '&gt;')
            formatted_text += f"  - **Simple LLM Summary:** {escaped_summary}\n"
        formatted_text += "\n"
        return formatted_text

    def update_markdown_file(self, filepath, papers, date_str):
        # ...实现不变...
        if not papers:
            print("没有论文需要添加")
            return

        # 只保留感兴趣的论文
        interested_papers = [paper for paper in papers if paper.get('is_interested', False)]

        existing_content = ""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_content = f.read()

        # 利用正则找到所有日期section
        date_section_pattern = re.compile(
            r"(^|\n)##\s*(\d{4}-\d{2}-\d{2}).*?(?=\n##\s|\Z)", re.DOTALL
        )
        all_sections = []
        for m in date_section_pattern.finditer(existing_content):
            section_start = m.start()
            section_content = m.group(0).lstrip('\n')
            section_date = m.group(2)
            all_sections.append((section_date, section_content, section_start))
        
        # 新section内容
        papers_content = f"## {date_str}\n\n"
        if interested_papers:
            for paper in interested_papers:
                papers_content += self.format_paper_with_enhanced_info(paper, date_str=date_str)
        else:
            print("No interesting papers for me today")
            papers_content += "No interesting papers for me today\n"

        replaced = False
        # 如有则替换当前日期section
        for idx, (dt, _, start_idx) in enumerate(all_sections):
            if dt == date_str:
                # 替换
                before = existing_content[:start_idx].rstrip('\n')
                after_idx = start_idx + len(_)
                after = existing_content[after_idx:]
                new_content = before
                if new_content and not new_content.endswith('\n'):
                    new_content += "\n"
                new_content += "\n" + papers_content
                if after and not after.startswith('\n'):
                    new_content += "\n"
                new_content += after.lstrip('\n')
                replaced = True
                print(f"日期 {date_str} 的内容已存在，已覆盖")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content.strip() + '\n')
                print(f"已将 {len(interested_papers)} 篇感兴趣的论文添加到文件: {filepath}")
                return

        # 如果没有，插入保持时间递增顺序（从小到大）
        # 找到插入点：第一个section日期大于本date_str，则插入在它前面；若找不到，追加到文件末尾
        insert_idx = None
        for idx, (dt, _, start_idx) in enumerate(all_sections):
            if dt > date_str:
                insert_idx = start_idx
                break
        if insert_idx is not None:
            # 插入到insert_idx前
            before = existing_content[:insert_idx].rstrip('\n')
            after = existing_content[insert_idx:]
            new_content = before
            if new_content and not new_content.endswith('\n'):
                new_content += "\n"
            new_content += "\n" + papers_content
            if after and not after.startswith('\n'):
                new_content += "\n"
            new_content += after.lstrip('\n')
            print(f"日期 {date_str} 的内容不存在，已按时间顺序插入")
        else:
            # 追加到最后
            new_content = existing_content.rstrip() + "\n\n" + papers_content
            print(f"日期 {date_str} 的内容不存在，已追加到最后")
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content.strip() + '\n')

        print(f"已将 {len(interested_papers)} 篇感兴趣的论文添加到文件: {filepath}")

    def find_or_create_weekly_file(self, date_str):
        """根据日期找到或创建对应的周文件"""
        week_range = self.get_week_range(date_str)
        if not week_range:
            return None
        
        filename = f"{week_range}.md"
        filepath = os.path.join(self.docs_daily_path, filename)
        
        if not os.path.exists(filepath):
            self.create_weekly_file(filepath, week_range)
        
        return filepath

    def create_weekly_file(self, filepath, week_range):
        """创建新的周文件"""
        start_date_str, end_date_str = week_range.split('-')
        start_date = datetime.strptime(start_date_str, '%Y%m%d')
        end_date = datetime.strptime(end_date_str, '%Y%m%d')
        
        content = f"""# {week_range}

"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"创建新的周文件: {filepath}")

    # ==================== 主处理流程 ====================
    
    def process_papers_by_date(self, target_date, categories=['cs.DC', 'cs.AI'], max_workers=2, max_papers=10):
        """
        根据指定日期处理论文的完整流程

        Args:
            target_date (str): 目标日期，格式为 'YYYY-MM-DD'
            categories (list): 论文分类列表
            max_workers (int): 并发处理数量
            max_papers (int): 最大处理论文数量（用于测试）
        """
        # 只支持单个日期
        if not (isinstance(target_date, str) and re.match(r'^\d{4}-\d{2}-\d{2}$', target_date)):
            print("target_date格式错误，只支持 'YYYY-MM-DD' 字符串")
            return

        print(f"开始处理日期: {target_date}")

        single_date = target_date
        print(f"\n==== 处理 {single_date} ====")
        # 1. 从arXiv获取论文
        print("步骤1: 从arXiv获取论文...")
        papers = self.fetch_arxiv_papers(categories=categories, max_results=1024, target_date=single_date)

        if not papers:
            print(f"日期 {single_date} 没有找到论文")
            return

        # 限制处理数量（用于测试）
        if max_papers and len(papers) > max_papers:
            papers = papers[:max_papers]
            print(f"限制处理前 {max_papers} 篇论文")

        print(f"找到 {len(papers)} 篇论文，开始处理...")

        # 2. 并发处理论文（下载PDF、调用LLM）
        print("步骤2: 处理论文（下载PDF、调用LLM）...")
        processed_papers = []

        for i, paper in enumerate(papers):
            print(f"{i+1}. {paper.get('title', 'N/A')}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self.process_single_paper, paper): paper 
                for paper in papers
            }

            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(future_to_paper), 
                             total=len(future_to_paper), desc="处理论文"):
                try:
                    processed_paper = future.result()
                    processed_papers.append(processed_paper)
                except Exception as e:
                    print(f"处理论文时出错: {e}")

        # 3. 统计结果
        interested_papers = [p for p in processed_papers if p.get('is_interested', False)]
        print(f"处理完成！总共 {len(processed_papers)} 篇论文，其中 {len(interested_papers)} 篇感兴趣")

        # 4. 更新markdown文件
        print("步骤3: 更新markdown文件...")
        weekly_file = self.find_or_create_weekly_file(single_date)
        if weekly_file:
            self.update_markdown_file(weekly_file, processed_papers, single_date)
            if interested_papers:
                print(f"处理完成！感兴趣的论文已添加到: {weekly_file}")
            else:
                print(f"处理完成！已添加日期记录到: {weekly_file}")
        else:
            print("无法创建或找到周文件")

def main():
    """
    主函数 - 使用示例
    """
    if not PDF_AVAILABLE:
        print("请先安装PyPDF2: pip install PyPDF2")
        return
    
    # 检查API密钥
    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("请设置DEEPSEEK_API_KEY环境变量")
        return
    
    # 创建处理器
    processor = CompletePaperProcessor()
    
    # 指定要处理的日期（只支持单天，格式'YYYY-MM-DD'）
    target_date = datetime.now().strftime('%Y-%m-%d') # 今天日期

    # 处理论文
    processor.process_papers_by_date(
        target_date=target_date,
        categories=['cs.DC', 'cs.AI', 'cs.LG'],  # 可以修改分类
        max_workers=10,  # 并发数量，建议不要太高
        max_papers=None    # 测试时限制论文数量，正式使用时可以设为None
    )

if __name__ == "__main__":
    main()