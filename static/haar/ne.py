from icrawler.builtin import BingImageCrawler         
classes=['trees','roads','Human faces']
number=100
for c in classes:
    bing_crawler=BingImageCrawler(storage={'root_dir':f'n/{c.replace(" ",".")}'}) #//see n is represent negaive images 
    bing_crawler.crawl(keyword=c,filters=None,max_num=number,offset=0)       