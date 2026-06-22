# VLM + SAM 2: поиск объектов по тексту на изображениях и видео

Colab-проект для портфолио: исследование пайплайна поиска и сегментации объектов по текстовому запросу на изображениях и видео.

Архитектура пайплайна:

- `Grounding DINO` ищет bounding boxes по тексту;
- `YOLO-World` используется как быстрый open-vocabulary baseline;
- `SAM 2` получает эти boxes и делает сегментацию;
- для видео берутся кадры, прогоняются через тот же пайплайн и собираются обратно в mp4;
- качество проверяется на автоматически собранных задачах из COCO val2017 через `mIoU`, `Precision@IoU=0.50`, `box_recall_50`, `task_mAP50`, pixel precision и pixel recall.

Это не production-система, а компактный исследовательский ноутбук: его можно запустить в Colab, поменять режим benchmark, подобрать prompt/thresholds и посмотреть, где находится bottleneck пайплайна.

## Как запустить

1. Открой `VLM_SAM2_text_object_search.ipynb` в Google Colab.
2. Включи GPU: `Runtime -> Change runtime type -> T4 GPU`.
3. Запускай ячейки сверху вниз.
4. Для своих файлов поменяй `TEXT_LABELS`, `IMAGE_URL` или загрузи файл через отдельную ячейку.

## Что получится

- картинка с найденными объектами, масками и подписями;
- таблица с найденными объектами и score;
- benchmark на COCO val2017 tasks;
- tune/test split, чтобы не подбирать параметры на финальной оценке;
- sweep параметров `box_threshold` и `text_threshold`;
- prompt-template tuning;
- сравнение `oracle_boxes_sam2`, `grounding_dino_sam2` и `yolo_world_sam2`;
- интерпретация результата метрик;
- короткий error analysis по лучшим и худшим примерам;
- короткое размеченное видео в `outputs/video_result.mp4`.

## Файлы

- `VLM_SAM2_text_object_search.ipynb` - основной ноутбук;
- `requirements_colab.txt` - зависимости, если хочется ставить руками;
- `.gitignore` - чтобы не тащить в GitHub тяжелые выходные файлы.

## Метрики

В ноутбуке считается компактная оценка качества:

- `mIoU` - среднее пересечение предсказанной маски с COCO-разметкой;
- `Precision@IoU=0.50` - доля примеров, где IoU не ниже 0.5;
- `box_recall_50` - насколько хорошо detector находит COCO boxes;
- `task_mAP50` - компактная AP@50 оценка для image-category задач;
- `pixel_precision` и `pixel_recall` - качество маски на уровне пикселей.

Это не полный официальный COCO leaderboard, но уже нормальный benchmark: задачи берутся из COCO val2017, параметры подбираются на tune split, финальное сравнение считается на test split.

## Режимы benchmark

В ноутбуке есть переменная `BENCH_MODE`:

- `debug` - быстрый прогон для проверки кода;
- `portfolio` - режим по умолчанию, примерно 100 test-задач;
- `fuller` - более тяжелый прогон, если есть время и GPU.

## Модели

В ноутбуке используются:

- `IDEA-Research/grounding-dino-tiny`
- `facebook/sam2.1-hiera-tiny`
- `yolov8s-world.pt`

Tiny/small-версии выбраны специально: они легче, быстрее стартуют в Colab и подходят для исследовательского проекта без долгой настройки окружения.

## Полезные ссылки

- MDETR, ICCV 2021 - https://openaccess.thecvf.com/content/ICCV2021/html/Kamath_MDETR_-_Modulated_Detection_for_End-to-End_Multi-Modal_Understanding_ICCV_2021_paper.html
- GLIP, CVPR 2022 - https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.pdf
- Segment Anything, ICCV 2023 - https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html
- YOLO-World, CVPR 2024 - https://arxiv.org/abs/2401.17270
- Grounding DINO - https://arxiv.org/abs/2303.05499
- Hugging Face docs: Grounding DINO - https://huggingface.co/docs/transformers/model_doc/grounding-dino
- Hugging Face docs: SAM 2 - https://huggingface.co/docs/transformers/model_doc/sam2
