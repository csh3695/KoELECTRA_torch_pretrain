import re

remove_news_keyword_list = [
    "gettyimagesbank",
    "gettyimageskorea",
    "게티이미지뱅크",
    "게티이미지코리아",
    "게티이미지",
    "※ 본 사이트에 게재되는 정보는 오류 및 지연이 있을 수 있으며, 그 이용에 따르는 책임은 이용자 본인에게 있습니다.",
    "news1",
    ":img_mark:",
    "창닫기기사",
]

remove_news_regex_list = [
    re.compile(
        r"(.{,4}·)*.{,4}(논설위원|문화팀 팀장|중국본부 팀장|국제경제팀 팀장|특파원|기자|인턴기자|선임기자|전문기자|여행전문기자|문화전문기자|금융전문기자|문화재전문기자|군사전문기자)\s?[A-Za-z0-9\.\+_-]+@[A-Za-z0-9\._-]+\.[a-zA-Z]*"
    ),
    re.compile(
        r"((.{,4}·)*.{,4}(논설위원|문화팀 팀장|중국본부 팀장|국제경제팀 팀장|특파원|기자|인턴기자|선임기자|전문기자|여행전문기자|문화전문기자|금융전문기자|문화재전문기자|군사전문기자))$"
    ),
    re.compile(r"무단\s?전재"),
    re.compile(r"재배포\s?금지"),
    re.compile(r"copyright"),
    re.compile(r"\([^\)]+=[^\)]+\)"),
    re.compile(r"\([^\)]+(기자|에디터)\)"),
]

remove_noise_regex_list = [
    re.compile(r"\[[^\[\]]{,50}\]"),
    re.compile(r"\{[^\{\}]{,50}\}"),
    re.compile(r"\<[^\<\>]{,50}\>"),
    re.compile(r"#([0-9a-zA-Z가-힣]*)"),
    re.compile(
        "((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"
    ),
    re.compile(r"[A-Za-z0-9\.\+_-]+@[A-Za-z0-9\._-]+\.[a-zA-Z]*"),
    re.compile(r"[^ .,?!/@$%~％·∼()\x00-\x7F가-힣一-龥\.]+"),
]

multi_space_regex = re.compile("\s{2,}")  # doublespace pattern -> " "
