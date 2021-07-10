from app.models import Response as ChatbotResponse
from app.models import Messages, Evaluations, TicketInfo
import json, ast

class topic_content_time_report:
    def get_topic_content_time_report(self, company_id):
        response_data_query_set = ChatbotResponse.objects.filter(company_id=company_id, parent_response_id=None)

        topic_dict = {}

        content_list = []
        content_dict = {}

        total_time = 0
        time_count = 0.00000001

        for response in response_data_query_set:
            evaluation_dict = ast.literal_eval(response.evaluation_data.replace('false','"false"'))

            evaluation_dict_body = evaluation_dict['body']

            for dic in evaluation_dict_body:
                if dic['is_impossible'] == False:
                    try:
                        topic = dic['topic']
                        if topic in topic_dict:
                            topic_dict[topic] += 1
                        else:
                            topic_dict[topic] = 1
                    except:
                        pass

                    try:
                        content = dic['content']
                        if content in content_list:
                            idx = content_list.index(content)
                            content_dict[idx] += 1
                        else:
                            idx = len(content_dict)
                            content_list.append(content)
                            content_dict[idx] = 1
                    except:
                        pass
                    
                    try:
                        total_time += dic['response_time']
                        time_count += 1
                    except:
                        pass

        return topic_dict, content_dict, content_list, total_time/time_count
