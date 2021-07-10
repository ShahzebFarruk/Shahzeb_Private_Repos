from app.models import Response as ChatbotResponse
from app.models import Messages, Evaluations, TicketInfo

class topic_content_report:
    def get_topic_content_report(self, company_id):
        response_data_query_set = ChatbotResponse.objects.filter(company_id=company_id, parent_response_id=None)

        topic_dict = {}

        content_list = []
        content_dict = {}

        for response in response_data_query_set:
            evaluation_dict = response.evaluation_data
            evaluation_dict_body = evaluation_dict['body']

            for dic in evaluation_dict_body:
                if dic['is_impossible'] == False:
                    topic = dic['topic']
                    content = dic['content']

                    if topic in topic_dict:
                        topic_dict[topic] += 1
                    else:
                        topic_dict[topic] = 1
                
                    if content in content_list:
                        idx = content_list.index(content)
                        content_dict[idx] += 1
                    else:
                        idx = len(content_dict)
                        content_list.append(content)
                        content_dict[idx] = 1

        return topic_dict, content_dict, content_list
