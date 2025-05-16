import re
import xml.etree.ElementTree as ET

class XMLArticle:
    @staticmethod
    def clean_text(text):
        if text is None:
            return ''
        # Replace special characters and tags
        text = re.sub(r'<.*?>', '', text)  # Remove tags like <break/>
        return text.replace('\u2028', ' ').replace('\u2010', '-').strip()

    @staticmethod
    def parse_table(table_root):
        sections = {}
        section_name = "unknown"
        cur_section = {}
        for tr in table_root.findall(".//tr"):
            cells = tr.findall("td")
            if len(cells) == 1 and int(cells[0].get("colspan",default="1")) > 1:
                content = [item.text for item in cells[0].iter() if re.findall(r'\w', item.text)]
                section = content[0] if len(content) > 0 else "unknown"
                if len(cur_section) > 0:
                    sections[section_name] = cur_section
                section_name = section
                cur_section = {}
            else:
                row_data = [XMLArticle.clean_text(ET.tostring(td, encoding='unicode', method='text')) for td in cells]
                if len(row_data) > 1:
                    cur_section[row_data[0]] = row_data[1:]
        if section_name:
            sections[section_name] = cur_section
        return sections

    def __init__(self, article_root, verbose=True, ref_type="doi", keep_fallbacks=True):
        self.root = article_root
        self.printv = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None
        self.ref_map = self.parse_references(ref_type)
        self.ref_type = ref_type
        self.keep_fallbacks = keep_fallbacks
    
    def parse_references(self, ref_type):
        ref_section = self.find_tag_by_child_content("References to studies included in this review", tag="ref-list")
        reference_map = {}
        n_ref = 0
        for ref_list in ref_section.iter("ref-list"):
            n_ref += 1
            ref_id = ref_list.get('id')
            reference_map[ref_id] = None
            for pub_id in ref_list.iter("pub-id"):
                if pub_id.get('pub-id-type', default="") == ref_type:
                    reference_map[ref_id] = pub_id.text
                    break
        self.printv(f'references parsed: {n_ref} studies found (with {len(reference_map)} total citations)')
        return reference_map


    def find_tag_by_child_content(self, content, match_case=False, tag="sec"):
        is_match = lambda text: (content in text) if match_case else (content.lower() in text.lower())
        all_tags = self.root.findall('.//'+tag)
        for tag in all_tags:
            for child in tag:
                txt = child.text 
                if txt is not None and is_match(txt):
                    return tag

    def parse_wrapped_table(self, table_wrap_root):
        ref_elem = table_wrap_root.find('.//xref')
        ref_id = ref_elem.get('rid')
        study_id = self.ref_map.get(ref_id)
        if (study_id is None):
            if self.keep_fallbacks:
                self.printv("warning: fallback to study name")
                study_id = f"NAME={ref_elem.text}"
            else:
                return None, None
        self.printv(f'parsing {study_id}', end=" -> ")
        table_root = table_wrap_root.find('.//table')
        table_data = XMLArticle.parse_table(table_root)
        self.printv(f'table data headers: {list(table_data.keys())}')
        return study_id, table_data
    
    def parse_included_studies(self):
        included_studies = self.find_tag_by_child_content("Characteristics of included studies")
        study_data = {}
        for wrapped_table in included_studies.iter('table-wrap'):
            study_id, table_data = self.parse_wrapped_table(wrapped_table)
            if study_id is None:
                continue
            study_data[study_id] = table_data
        return study_data

def parse_xml_batch(raw_response):
    return ET.fromstring(raw_response)