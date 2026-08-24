from bs4 import BeautifulSoup

from spacex_capstone.scraping import (
    extract_column_from_header,
    find_launch_tables,
    parse_payload_mass,
)


class TestExtractColumnFromHeader:
    def test_strips_line_break_and_joins_with_space(self):
        th = BeautifulSoup("<th>Launch<br/>site</th>", "html.parser").th
        assert extract_column_from_header(th) == "Launch site"

    def test_strips_reference_superscript(self):
        th = BeautifulSoup("<th>Payload<sup>[a]</sup></th>", "html.parser").th
        assert extract_column_from_header(th) == "Payload"

    def test_link_wrapped_header_loses_its_text(self):
        # Known, pre-existing behavior (matches notebook 02): a header whose
        # entire text is inside an <a> tag gets stripped along with the link,
        # since row.a.extract() removes the link and everything inside it.
        # This is exactly why notebook 08 uses "Launch outcome" (a plain-text
        # header) rather than "Booster landing" (a linked one) to identify
        # launch-record tables.
        th = BeautifulSoup('<th><a href="/wiki/X">Booster<br/>landing</a></th>', "html.parser").th
        assert extract_column_from_header(th) == ""

    def test_purely_numeric_header_returns_none(self):
        th = BeautifulSoup("<th>42</th>", "html.parser").th
        assert extract_column_from_header(th) is None


class TestParsePayloadMass:
    def test_extracts_kg_value(self):
        td = BeautifulSoup("<td>15,600 kg</td>", "html.parser").td
        assert parse_payload_mass(td) == 15600.0

    def test_blank_cell_returns_zero(self):
        td = BeautifulSoup("<td></td>", "html.parser").td
        assert parse_payload_mass(td) == 0.0

    def test_cell_without_kg_returns_zero(self):
        td = BeautifulSoup("<td>unknown</td>", "html.parser").td
        assert parse_payload_mass(td) == 0.0


class TestFindLaunchTables:
    def test_matches_tables_with_both_required_classes_regardless_of_order(self):
        html = """
        <table class="wikitable plainrowheaders collapsible sticky-header"><tr><td>a</td></tr></table>
        <table class="wikitable sticky-header collapsible plainrowheaders"><tr><td>b</td></tr></table>
        <table class="wikitable"><tr><td>c</td></tr></table>
        <table class="wikitable plainrowheaders"><tr><td>d</td></tr></table>
        """
        soup = BeautifulSoup(html, "html.parser")
        matched = find_launch_tables(soup)
        assert len(matched) == 2
