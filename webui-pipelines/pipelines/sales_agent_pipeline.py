from typing import List, Union, Generator, Iterator
import requests
import json

from subprocess import call


class Pipeline:
    def __init__(self):
        self.name = "AppleScript Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        

        #напишите мне пожалуйста на popset26@gmail.com
        # Какие самые негативные темы обсуждают пользовтаели
        # "напишите мне пожалуйста на asck.petrov@yandex.ru"
        # удалите мой контакт из вашей рассылки
        # "Расскажи мне про вкусный кофе"
        # URL вашего API
        url = "http://localhost:8002/query"

        # Данные, которые мы отправляем в запросе
        data = {
            "answer": user_message
        }

        # Отправляем POST-запрос
        response = requests.post(url, json=data)

        # Проверяем, успешен ли запрос
        if response.status_code == 200:
            # Если успешен, выводим результат
            result = response.json()
            return(result['result'])
        else:
            # Если произошла ошибка, выводим сообщение
            return(f"Ошибка {response.status_code}: {response.text}")
