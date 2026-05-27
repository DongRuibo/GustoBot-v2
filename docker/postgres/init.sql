-- GustoBot-v2 PostgreSQL 初始化脚本。
-- 目标：统一承载 KB/pgvector、Text2SQL 业务表、Schema Catalog、评估日志和 trace 元数据。

CREATE EXTENSION IF NOT EXISTS vector;

-- KB RAG：文档级元数据。
CREATE TABLE IF NOT EXISTS kb_documents (
    document_id text PRIMARY KEY,
    title text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- KB RAG：chunk + embedding。Docker 默认 embedding 维度为 1024。
CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES kb_documents(document_id) ON DELETE CASCADE,
    content text NOT NULL,
    embedding vector(1024) NOT NULL,
    search_text text NOT NULL DEFAULT '',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_metadata ON kb_chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_search_text
    ON kb_chunks USING gin (to_tsvector('simple', COALESCE(NULLIF(search_text, ''), content)));

-- Text2SQL：示例业务结构化表。后续真实业务表也放在 PostgreSQL 中。
CREATE TABLE IF NOT EXISTS recipes (
    recipe_id integer PRIMARY KEY,
    name text NOT NULL,
    cuisine text NOT NULL,
    difficulty text NOT NULL,
    cooking_time_minutes integer NOT NULL,
    popularity integer NOT NULL,
    created_year integer NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

INSERT INTO recipes
    (recipe_id, name, cuisine, difficulty, cooking_time_minutes, popularity, created_year, metadata)
VALUES
    (1, '宫保鸡丁', '川菜', '中等', 25, 96, 2024, '{"source":"docker_seed"}'),
    (2, '麻婆豆腐', '川菜', '简单', 18, 94, 2024, '{"source":"docker_seed"}'),
    (3, '鱼香肉丝', '川菜', '中等', 30, 88, 2025, '{"source":"docker_seed"}'),
    (4, '白灼虾', '粤菜', '简单', 12, 82, 2025, '{"source":"docker_seed"}'),
    (5, '叉烧', '粤菜', '中等', 90, 89, 2026, '{"source":"docker_seed"}'),
    (6, '佛跳墙', '闽菜', '困难', 180, 91, 2026, '{"source":"docker_seed"}'),
    (7, '白菜炖豆腐', '家常菜', '简单', 20, 76, 2026, '{"source":"docker_seed"}')
ON CONFLICT (recipe_id) DO UPDATE
SET name = EXCLUDED.name,
    cuisine = EXCLUDED.cuisine,
    difficulty = EXCLUDED.difficulty,
    cooking_time_minutes = EXCLUDED.cooking_time_minutes,
    popularity = EXCLUDED.popularity,
    created_year = EXCLUDED.created_year,
    metadata = EXCLUDED.metadata;

-- Schema Catalog：Text2SQL 只从这里检索允许暴露给生成器的表结构。
CREATE TABLE IF NOT EXISTS schema_catalog (
    table_name text PRIMARY KEY,
    table_comment text NOT NULL,
    business_meaning text NOT NULL,
    module text NOT NULL,
    columns jsonb NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO schema_catalog
    (table_name, table_comment, business_meaning, module, columns, metadata)
VALUES (
    'recipes',
    '菜谱主表，保存菜名、菜系、难度、耗时和热度等结构化字段。',
    '用于回答菜谱数量统计、菜系统计、排名、平均耗时和趋势类问题。',
    'recipe_analytics',
    '[
        {"name":"recipe_id","data_type":"integer","comment":"菜谱唯一编号","sample_values":[1,2,3]},
        {"name":"name","data_type":"text","comment":"菜谱名称","sample_values":["宫保鸡丁","麻婆豆腐"]},
        {"name":"cuisine","data_type":"text","comment":"菜系名称","sample_values":["川菜","粤菜","闽菜"]},
        {"name":"difficulty","data_type":"text","comment":"制作难度","sample_values":["简单","中等","困难"]},
        {"name":"cooking_time_minutes","data_type":"integer","comment":"烹饪耗时，单位分钟","sample_values":[15,25,90]},
        {"name":"popularity","data_type":"integer","comment":"热度分，用于排名分析","sample_values":[82,95,76]},
        {"name":"created_year","data_type":"integer","comment":"菜谱录入年份，用于趋势分析","sample_values":[2024,2025,2026]}
    ]'::jsonb,
    '{"source":"docker_seed"}'
)
ON CONFLICT (table_name) DO UPDATE
SET table_comment = EXCLUDED.table_comment,
    business_meaning = EXCLUDED.business_meaning,
    module = EXCLUDED.module,
    columns = EXCLUDED.columns,
    metadata = EXCLUDED.metadata,
    updated_at = now();

-- 食品数据底座：Open Food Facts / USDA 清洗后的商品主表。
CREATE TABLE IF NOT EXISTS food_products (
    product_id text PRIMARY KEY,
    name text NOT NULL,
    brand text,
    category text NOT NULL,
    country text,
    ingredients_text text,
    allergens text,
    energy_kcal numeric(12,4),
    protein numeric(12,4),
    fat numeric(12,4),
    carbohydrates numeric(12,4),
    sugars numeric(12,4),
    salt numeric(12,4),
    source text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS food_nutrients (
    product_id text NOT NULL REFERENCES food_products(product_id) ON DELETE CASCADE,
    nutrient_name text NOT NULL,
    value numeric(12,4) NOT NULL,
    unit text NOT NULL,
    source text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (product_id, nutrient_name)
);

CREATE INDEX IF NOT EXISTS idx_food_products_category ON food_products(category);
CREATE INDEX IF NOT EXISTS idx_food_products_brand ON food_products(brand);
CREATE INDEX IF NOT EXISTS idx_food_products_source ON food_products(source);

INSERT INTO schema_catalog
    (table_name, table_comment, business_meaning, module, columns, metadata)
VALUES (
    'food_products',
    '食品商品主表，保存商品名称、品牌、分类、配料、过敏原和常用营养标签。',
    '用于食品商品筛选、品牌统计、分类统计、糖分/蛋白质/能量排序和营养标签查询。',
    'food_analytics',
    '[
        {"name":"product_id","data_type":"text","comment":"商品唯一编号，带 source 前缀","sample_values":["off:3017620422003","usda:1104067"]},
        {"name":"name","data_type":"text","comment":"商品或标准食物名称","sample_values":["Peanut Protein Bar","燕麦奶"]},
        {"name":"brand","data_type":"text","comment":"品牌","sample_values":["Demo Foods"]},
        {"name":"category","data_type":"text","comment":"食品分类","sample_values":["Protein bars","Plant-based drinks"]},
        {"name":"country","data_type":"text","comment":"国家或地区","sample_values":["United States","China"]},
        {"name":"ingredients_text","data_type":"text","comment":"配料文本","sample_values":["peanuts, soy protein isolate"]},
        {"name":"allergens","data_type":"text","comment":"过敏原文本","sample_values":["peanuts, soy"]},
        {"name":"energy_kcal","data_type":"numeric","comment":"每 100g 能量 kcal","sample_values":[58,420]},
        {"name":"protein","data_type":"numeric","comment":"每 100g 蛋白质 g","sample_values":[1.2,28]},
        {"name":"fat","data_type":"numeric","comment":"每 100g 脂肪 g","sample_values":[1.5,16]},
        {"name":"carbohydrates","data_type":"numeric","comment":"每 100g 碳水 g","sample_values":[9,42]},
        {"name":"sugars","data_type":"numeric","comment":"每 100g 糖 g","sample_values":[3.2,18]},
        {"name":"salt","data_type":"numeric","comment":"每 100g 盐 g","sample_values":[0.12,0.6]},
        {"name":"source","data_type":"text","comment":"数据来源","sample_values":["openfoodfacts","usda"]}
    ]'::jsonb,
    '{"source":"food_dataset_init"}'
)
ON CONFLICT (table_name) DO UPDATE
SET table_comment = EXCLUDED.table_comment,
    business_meaning = EXCLUDED.business_meaning,
    module = EXCLUDED.module,
    columns = EXCLUDED.columns,
    metadata = EXCLUDED.metadata,
    updated_at = now();

INSERT INTO schema_catalog
    (table_name, table_comment, business_meaning, module, columns, metadata)
VALUES (
    'food_nutrients',
    '食品营养素长表，每个商品每种营养素一行。',
    '用于扩展营养素明细、按营养素名称过滤和与 food_products 关联分析。',
    'food_analytics',
    '[
        {"name":"product_id","data_type":"text","comment":"商品唯一编号","sample_values":["off:3017620422003"]},
        {"name":"nutrient_name","data_type":"text","comment":"营养素名称","sample_values":["protein","sugars"]},
        {"name":"value","data_type":"numeric","comment":"营养素值","sample_values":[3.2,28]},
        {"name":"unit","data_type":"text","comment":"单位","sample_values":["g","kcal"]},
        {"name":"source","data_type":"text","comment":"数据来源","sample_values":["openfoodfacts","usda"]}
    ]'::jsonb,
    '{"source":"food_dataset_init"}'
)
ON CONFLICT (table_name) DO UPDATE
SET table_comment = EXCLUDED.table_comment,
    business_meaning = EXCLUDED.business_meaning,
    module = EXCLUDED.module,
    columns = EXCLUDED.columns,
    metadata = EXCLUDED.metadata,
    updated_at = now();

-- 评估日志：后续 scripts/evaluate.py 可选择写入这里。
CREATE TABLE IF NOT EXISTS evaluation_logs (
    id bigserial PRIMARY KEY,
    run_id text NOT NULL,
    sample_id text,
    route_expected text,
    route_actual text,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- trace 元数据：当前仍默认写 JSONL，这张表作为后续结构化观测落点。
CREATE TABLE IF NOT EXISTS trace_events (
    id bigserial PRIMARY KEY,
    trace_id text NOT NULL,
    event_type text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_trace_events_trace_id ON trace_events(trace_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_logs_run_id ON evaluation_logs(run_id);

-- GraphRAG 食材 taxonomy：生产环境以 PostgreSQL 为源数据，YAML seed 只负责初始化。
CREATE TABLE IF NOT EXISTS ingredient_categories (
    slug text PRIMARY KEY,
    name text NOT NULL,
    enabled boolean NOT NULL DEFAULT true,
    priority integer NOT NULL DEFAULT 100,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingredient_category_aliases (
    category_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
    alias text NOT NULL,
    enabled boolean NOT NULL DEFAULT true,
    priority integer NOT NULL DEFAULT 100,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (category_slug, alias)
);

CREATE TABLE IF NOT EXISTS ingredient_category_patterns (
    id bigserial PRIMARY KEY,
    category_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
    name_pattern text,
    source_category_pattern text,
    pattern_key text GENERATED ALWAYS AS (coalesce(name_pattern, '') || '|' || coalesce(source_category_pattern, '')) STORED,
    enabled boolean NOT NULL DEFAULT true,
    priority integer NOT NULL DEFAULT 100,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (name_pattern IS NOT NULL OR source_category_pattern IS NOT NULL),
    UNIQUE (category_slug, pattern_key)
);

CREATE TABLE IF NOT EXISTS ingredient_category_hierarchy (
    child_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
    parent_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
    enabled boolean NOT NULL DEFAULT true,
    priority integer NOT NULL DEFAULT 100,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (child_slug, parent_slug)
);

CREATE TABLE IF NOT EXISTS ingredient_category_assignments (
    id bigserial PRIMARY KEY,
    ingredient_name text,
    source_category text,
    category_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
    assignment_key text GENERATED ALWAYS AS (coalesce(ingredient_name, '') || '|' || coalesce(source_category, '') || '|' || category_slug) STORED,
    enabled boolean NOT NULL DEFAULT true,
    priority integer NOT NULL DEFAULT 0,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (ingredient_name IS NOT NULL OR source_category IS NOT NULL),
    UNIQUE (assignment_key)
);

-- 应用会话：用于前端会话列表和历史消息恢复。
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id text PRIMARY KEY,
    user_id text,
    title text NOT NULL,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    message_id text PRIMARY KEY,
    session_id text NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role text NOT NULL,
    content text NOT NULL,
    route_type text,
    trace_id text,
    evidences jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    order_index integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_session_snapshots (
    snapshot_id text PRIMARY KEY,
    session_id text NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    message_id text NOT NULL REFERENCES chat_messages(message_id) ON DELETE CASCADE,
    trace_id text,
    route_type text,
    answer text NOT NULL,
    evidences jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, order_index);
CREATE INDEX IF NOT EXISTS idx_chat_session_snapshots_session ON chat_session_snapshots(session_id, created_at DESC);

-- 上传文件登记：业务链路只通过 upload://file_id 读取已登记文件。
CREATE TABLE IF NOT EXISTS uploaded_files (
    file_id text PRIMARY KEY,
    kind text NOT NULL,
    original_name text NOT NULL,
    stored_name text NOT NULL,
    relative_path text NOT NULL,
    content_type text,
    size_bytes bigint NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    deleted_at timestamptz
);

CREATE INDEX IF NOT EXISTS idx_uploaded_files_kind ON uploaded_files(kind);

-- 真实菜谱业务表：MySQL 旧 SQL 只作为数据来源，运行时统一落到 PostgreSQL。
CREATE TABLE IF NOT EXISTS recipe_cuisines (
    id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE,
    cooking_style text,
    typical_tools jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recipe_tools (
    id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE,
    type text NOT NULL DEFAULT 'other',
    material text,
    capacity text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recipe_ingredients_master (
    id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE,
    category text,
    calories numeric(10,2) NOT NULL DEFAULT 0,
    protein numeric(10,2) NOT NULL DEFAULT 0,
    carbs numeric(10,2) NOT NULL DEFAULT 0,
    fat numeric(10,2) NOT NULL DEFAULT 0,
    storage_method text,
    shelf_life integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recipe_records (
    id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE,
    description text,
    image_url text,
    video_url text,
    total_time integer NOT NULL DEFAULT 0,
    servings integer NOT NULL DEFAULT 4,
    difficulty text NOT NULL DEFAULT 'easy',
    cuisine_id integer REFERENCES recipe_cuisines(id) ON DELETE SET NULL,
    total_calories numeric(10,2) NOT NULL DEFAULT 0,
    total_protein numeric(10,2) NOT NULL DEFAULT 0,
    total_carbs numeric(10,2) NOT NULL DEFAULT 0,
    total_fat numeric(10,2) NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recipe_steps (
    id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    recipe_id integer NOT NULL REFERENCES recipe_records(id) ON DELETE CASCADE,
    step_number integer NOT NULL,
    action text NOT NULL,
    instruction text NOT NULL,
    duration integer NOT NULL DEFAULT 0,
    temperature text,
    tools_used jsonb NOT NULL DEFAULT '[]'::jsonb,
    tips text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (recipe_id, step_number)
);

CREATE TABLE IF NOT EXISTS recipe_ingredients (
    id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    recipe_id integer NOT NULL REFERENCES recipe_records(id) ON DELETE CASCADE,
    ingredient_id integer NOT NULL REFERENCES recipe_ingredients_master(id) ON DELETE RESTRICT,
    quantity text NOT NULL,
    unit text,
    prep_method text,
    prep_time integer NOT NULL DEFAULT 0,
    is_main boolean NOT NULL DEFAULT false,
    substitute text,
    adjusted_calories numeric(10,2) NOT NULL DEFAULT 0,
    ingredient_type text NOT NULL DEFAULT 'auxiliary',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (recipe_id, ingredient_id, ingredient_type)
);

CREATE TABLE IF NOT EXISTS recipe_step_tools (
    id integer GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    step_id integer NOT NULL REFERENCES recipe_steps(id) ON DELETE CASCADE,
    tool_id integer NOT NULL REFERENCES recipe_tools(id) ON DELETE RESTRICT,
    usage_text text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (step_id, tool_id)
);
