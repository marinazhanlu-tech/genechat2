"""
NCBI基因数据库处理器

专门处理NCBI基因数据，实现论文中的三胞胎数据格式：
(dna_sequence, prompt, target_description)

关键特性：
- 从NCBI下载和处理基因数据
- 支持超长DNA序列（160kb）
- 生成基因功能描述
- 数据质量验证和清洗
- 符合论文数据处理流程
"""

import os
import json
import requests
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import re
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class GeneRecord:
    """基因记录数据结构"""
    gene_id: str
    gene_symbol: str
    organism: str
    dna_sequence: str
    sequence_length: int
    chromosome: Optional[str] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    strand: Optional[str] = None
    gene_function: Optional[str] = None
    gene_description: Optional[str] = None
    go_terms: Optional[List[str]] = None
    protein_domains: Optional[List[str]] = None
    gene_family: Optional[str] = None
    synonyms: Optional[List[str]] = None

@dataclass
class GeneTriplet:
    """论文中的三胞胎数据格式"""
    dna_sequence: str
    prompt: str
    target_description: str
    gene_metadata: Dict

@dataclass
class NCBIConfig:
    """NCBI数据处理器配置"""
    email: str = "your_email@example.com"  # NCBI API需要
    api_key: Optional[str] = None
    max_retries: int = 3
    request_delay: float = 0.5  # 请求间隔，避免被限制
    batch_size: int = 100
    max_sequence_length: int = 160000  # 160kb
    min_sequence_length: int = 1000    # 1kb最小值
    cache_dir: str = "./cache/ncbi"
    validate_dna: bool = True
    include_introns: bool = True
    download_genomic: bool = True

class NCBIGeneProcessor:
    """NCBI基因数据库处理器"""

    def __init__(self, config: NCBIConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 设置NCBI API
        Entrez.email = config.email
        if config.api_key:
            Entrez.api_key = config.api_key

        logger.info(f"NCBI基因处理器初始化完成，邮箱: {config.email}")

    def download_gene_summary(self, gene_ids: List[str]) -> Dict[str, Dict]:
        """下载基因摘要信息"""
        logger.info(f"开始下载 {len(gene_ids)} 个基因的摘要信息...")

        gene_summaries = {}
        failed_ids = []

        # 分批处理，避免API限制
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(gene_ids), batch_size), desc="下载基因摘要"):
            batch_ids = gene_ids[i:i+batch_size]

            try:
                # 获取基因摘要
                handle = Entrez.esummary(db="gene", id=",".join(batch_ids))
                records = Entrez.read(handle)
                handle.close()

                for record in records["DocumentSummarySet"]["DocumentSummary"]:
                    gene_id = record["uid"]
                    gene_summaries[gene_id] = record

                # API限制延迟
                import time
                time.sleep(self.config.request_delay)

            except Exception as e:
                logger.error(f"下载批次 {i//batch_size} 失败: {e}")
                failed_ids.extend(batch_ids)

        if failed_ids:
            logger.warning(f"下载失败的基因ID: {len(failed_ids)}")

        logger.info(f"成功下载 {len(gene_summaries)} 个基因摘要")
        return gene_summaries

    def download_gene_sequences(self, gene_ids: List[str], organism: str = "human") -> Dict[str, str]:
        """下载基因序列"""
        logger.info(f"开始下载 {len(gene_ids)} 个基因的序列...")

        gene_sequences = {}
        failed_ids = []

        for gene_id in tqdm(gene_ids, desc="下载基因序列"):
            try:
                # 获取基因组信息
                handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
                gene_xml = handle.read()
                handle.close()

                # 解析XML获取genomic accession
                accession = self._extract_genomic_accession(gene_xml)
                if not accession:
                    logger.warning(f"基因 {gene_id} 找不到genomic accession")
                    failed_ids.append(gene_id)
                    continue

                # 下载基因组序列
                seq_handle = Entrez.efetch(
                    db="nucleotide",
                    id=accession,
                    rettype="fasta",
                    retmode="text"
                )
                fasta_data = seq_handle.read()
                seq_handle.close()

                # 解析FASTA
                from io import StringIO
                seq_record = SeqIO.read(StringIO(fasta_data), "fasta")
                gene_sequences[gene_id] = str(seq_record.seq)

                # API限制延迟
                import time
                time.sleep(self.config.request_delay)

            except Exception as e:
                logger.error(f"下载基因 {gene_id} 序列失败: {e}")
                failed_ids.append(gene_id)

        if failed_ids:
            logger.warning(f"下载序列失败的基因ID: {len(failed_ids)}")

        logger.info(f"成功下载 {len(gene_sequences)} 个基因序列")
        return gene_sequences

    def _extract_genomic_accession(self, gene_xml: str) -> Optional[str]:
        """从基因XML中提取genomic accession"""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(gene_xml)

            # 查找genomic accession
            for elem in root.iter():
                if elem.tag == "GenomicInfoType":
                    for subelem in elem:
                        if subelem.tag == "ChrAccVer":
                            return subelem.text

            return None
        except Exception as e:
            logger.error(f"解析XML失败: {e}")
            return None

    def process_single_gene(self, gene_id: str, gene_summary: Dict,
                            gene_sequence: str) -> Optional[GeneRecord]:
        """处理单个基因记录"""
        try:
            # 提取基本信息
            gene_symbol = gene_summary.get("name", "")
            organism = gene_summary.get("organism", {}).get("scientific", "unknown")

            # 验证DNA序列
            if self.config.validate_dna:
                if not self._validate_dna_sequence(gene_sequence):
                    logger.warning(f"基因 {gene_id} DNA序列无效")
                    return None

            # 截断序列长度
            if len(gene_sequence) > self.config.max_sequence_length:
                gene_sequence = gene_sequence[:self.config.max_sequence_length]
                logger.info(f"基因 {gene_id} 序列截断到 {len(gene_sequence)} bp")

            if len(gene_sequence) < self.config.min_sequence_length:
                logger.warning(f"基因 {gene_id} 序列太短: {len(gene_sequence)} bp")
                return None

            # 提取功能描述
            description = gene_summary.get("description", "")
            summary = gene_summary.get("summary", "")

            # 提取GO术语
            go_terms = self._extract_go_terms(gene_summary)

            # 提取基因位置信息
            chrom_info = self._extract_chromosomal_info(gene_summary)

            # 创建基因记录
            gene_record = GeneRecord(
                gene_id=gene_id,
                gene_symbol=gene_symbol,
                organism=organism,
                dna_sequence=gene_sequence,
                sequence_length=len(gene_sequence),
                chromosome=chrom_info.get("chromosome"),
                start_position=chrom_info.get("start"),
                end_position=chrom_info.get("end"),
                strand=chrom_info.get("strand"),
                gene_function=summary,
                gene_description=description,
                go_terms=go_terms,
                synonyms=gene_summary.get("otherdesignations", [])
            )

            return gene_record

        except Exception as e:
            logger.error(f"处理基因 {gene_id} 失败: {e}")
            return None

    def _validate_dna_sequence(self, sequence: str) -> bool:
        """验证DNA序列"""
        if not sequence:
            return False

        # 允许的DNA碱基
        valid_bases = set('ACGTN')
        return all(base.upper() in valid_bases for base in sequence)

    def _extract_go_terms(self, gene_summary: Dict) -> List[str]:
        """提取GO术语"""
        go_terms = []
        try:
            # 从摘要中提取GO术语
            summary = gene_summary.get("summary", "")
            go_pattern = r'GO:\d{7}'
            go_matches = re.findall(go_pattern, summary)
            go_terms.extend(go_matches)

            # 从其他字段提取
            if "properties" in gene_summary:
                properties = gene_summary["properties"]
                for prop in properties:
                    if prop.get("name") == "GeneOntology":
                        go_terms.extend(prop.get("value", "").split(";"))

        except Exception as e:
            logger.debug(f"提取GO术语失败: {e}")

        return list(set(go_terms))  # 去重

    def _extract_chromosomal_info(self, gene_summary: Dict) -> Dict:
        """提取染色体信息"""
        chrom_info = {}
        try:
            if "chromosome" in gene_summary:
                chrom_info["chromosome"] = gene_summary["chromosome"]

            if "maplocation" in gene_summary:
                map_location = gene_summary["maplocation"]
                # 尝试解析位置信息
                if "p" in map_location or "q" in map_location:
                    chrom_info["map_location"] = map_location

            # 从genomic info提取位置
            if "genomicinfo" in gene_summary:
                genomic_info = gene_summary["genomicinfo"][0]  # 通常取第一个
                chrom_info["chromosome"] = genomic_info.get("chraccver", "")
                chrom_info["start"] = int(genomic_info.get("chrstart", 0))
                chrom_info["end"] = int(genomic_info.get("chrend", 0))

        except Exception as e:
            logger.debug(f"提取染色体信息失败: {e}")

        return chrom_info

    def create_gene_triplet(self, gene_record: GeneRecord) -> GeneTriplet:
        """创建论文中的三胞胎数据格式"""
        # DNA序列
        dna_sequence = gene_record.dna_sequence

        # 固定提示（论文中使用的确切提示）
        prompt = "please predict the function of this gene"

        # 生成目标描述
        target_description = self._generate_gene_description(gene_record)

        # 元数据
        gene_metadata = asdict(gene_record)
        del gene_metadata["dna_sequence"]  # 不在元数据中重复存储

        return GeneTriplet(
            dna_sequence=dna_sequence,
            prompt=prompt,
            target_description=target_description,
            gene_metadata=gene_metadata
        )

    def _generate_gene_description(self, gene_record: GeneRecord) -> str:
        """生成基因功能描述（论文中的输出格式）"""
        parts = []

        # 生物体信息
        parts.append(f"This gene belongs to {gene_record.organism}")

        # 位置信息
        if gene_record.chromosome:
            location_parts = []
            location_parts.append(f"chromosome {gene_record.chromosome}")

            if gene_record.start_position and gene_record.end_position:
                location_parts.append(f"positions {gene_record.start_position}-{gene_record.end_position}")

            if gene_record.strand:
                location_parts.append(f"{gene_record.strand} strand")

            parts.append(f"located in {', '.join(location_parts)}")

        # 功能描述
        if gene_record.gene_function:
            parts.append(f"functions include {gene_record.gene_function.lower()}")

        # 基因符号
        if gene_record.gene_symbol:
            parts.append(f"also known as {gene_record.gene_symbol}")

        # GO术语
        if gene_record.go_terms:
            go_count = min(5, len(gene_record.go_terms))  # 限制GO术语数量
            go_terms_list = gene_record.go_terms[:go_count]
            parts.append(f"associated with gene ontology terms: {', '.join(go_terms_list)}")

        # 组合描述
        description = " ".join(parts) + "."

        return description

    def process_gene_dataset(self, gene_ids: List[str],
                           save_dir: Optional[str] = None) -> List[GeneTriplet]:
        """处理整个基因数据集"""
        logger.info(f"开始处理 {len(gene_ids)} 个基因...")

        # 检查缓存
        cache_file = self.cache_dir / f"gene_triplets_{len(gene_ids)}.json"
        if cache_file.exists():
            logger.info("从缓存加载基因三胞胎数据...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            return [GeneTriplet(**triplet) for triplet in cached_data]

        # 下载基因摘要
        gene_summaries = self.download_gene_summary(gene_ids)

        # 下载基因序列
        gene_sequences = self.download_gene_sequences(gene_ids)

        # 处理每个基因
        gene_triplets = []
        failed_genes = []

        for gene_id in tqdm(gene_ids, desc="处理基因"):
            if gene_id not in gene_summaries:
                logger.warning(f"基因 {gene_id} 没有摘要信息")
                failed_genes.append(gene_id)
                continue

            if gene_id not in gene_sequences:
                logger.warning(f"基因 {gene_id} 没有序列数据")
                failed_genes.append(gene_id)
                continue

            # 处理单个基因
            gene_record = self.process_single_gene(
                gene_id,
                gene_summaries[gene_id],
                gene_sequences[gene_id]
            )

            if gene_record is None:
                failed_genes.append(gene_id)
                continue

            # 创建三胞胎数据
            triplet = self.create_gene_triplet(gene_record)
            gene_triplets.append(triplet)

        logger.info(f"成功处理 {len(gene_triplets)} 个基因，失败 {len(failed_genes)} 个")

        # 保存到缓存
        if save_dir or self.cache_dir:
            save_path = save_dir or cache_file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(triplet) for triplet in gene_triplets], f, indent=2)
            logger.info(f"基因三胞胎数据已保存到: {save_path}")

        return gene_triplets

    def get_human_gene_ids(self, limit: Optional[int] = None) -> List[str]:
        """获取人类基因ID列表"""
        logger.info("获取人类基因ID列表...")

        try:
            # 搜索人类基因
            handle = Entrez.esearch(
                db="gene",
                term=f"Homo sapiens[Organism]",
                retmax=limit or 60000  # 人类基因数量
            )
            record = Entrez.read(handle)
            handle.close()

            gene_ids = record["IdList"]
            logger.info(f"找到 {len(gene_ids)} 个人类基因")

            return gene_ids

        except Exception as e:
            logger.error(f"获取人类基因ID失败: {e}")
            return []

    def download_comprehensive_dataset(self,
                                   organism: str = "Homo sapiens",
                                   limit: Optional[int] = 1000,
                                   output_dir: str = "./data/ncbi_genes") -> List[GeneTriplet]:
        """下载完整的NCBI基因数据集"""
        logger.info(f"开始下载 {organism} 基因数据集，限制: {limit}")

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取基因ID
        if organism.lower() == "homo sapiens" or organism.lower() == "human":
            gene_ids = self.get_human_gene_ids(limit)
        else:
            # 其他物种的处理逻辑
            logger.warning(f"物种 {organism} 的基因ID获取尚未实现")
            return []

        if not gene_ids:
            logger.error("没有找到基因ID")
            return []

        # 处理基因数据集
        gene_triplets = self.process_gene_dataset(gene_ids)

        # 保存详细数据
        save_path = output_path / f"gene_dataset_{len(gene_triplets)}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(triplet) for triplet in gene_triplets], f, indent=2)

        # 保存统计信息
        self._save_dataset_statistics(gene_triplets, output_path)

        logger.info(f"基因数据集下载完成，保存到: {output_path}")
        return gene_triplets

    def _save_dataset_statistics(self, gene_triplets: List[GeneTriplet], output_path: Path):
        """保存数据集统计信息"""
        stats = {
            "total_genes": len(gene_triplets),
            "sequence_length_stats": {},
            "organism_distribution": {},
            "average_description_length": 0,
            "go_terms_distribution": {}
        }

        sequence_lengths = []
        description_lengths = []
        organisms = {}
        go_terms_count = []

        for triplet in gene_triplets:
            # 序列长度统计
            seq_length = len(triplet.dna_sequence)
            sequence_lengths.append(seq_length)

            # 描述长度统计
            desc_length = len(triplet.target_description)
            description_lengths.append(desc_length)

            # 生物体分布
            organism = triplet.gene_metadata.get("organism", "unknown")
            organisms[organism] = organisms.get(organism, 0) + 1

            # GO术语数量
            go_terms = triplet.gene_metadata.get("go_terms", [])
            go_terms_count.append(len(go_terms))

        # 计算统计信息
        stats["sequence_length_stats"] = {
            "min": min(sequence_lengths),
            "max": max(sequence_lengths),
            "mean": np.mean(sequence_lengths),
            "std": np.std(sequence_lengths),
            "median": np.median(sequence_lengths)
        }

        stats["organism_distribution"] = organisms
        stats["average_description_length"] = np.mean(description_lengths)
        stats["go_terms_distribution"] = {
            "min": min(go_terms_count),
            "max": max(go_terms_count),
            "mean": np.mean(go_terms_count)
        }

        # 保存统计信息
        with open(output_path / "dataset_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"数据集统计信息: {stats}")

# 便捷函数
def create_ncbi_processor(email: str,
                         api_key: Optional[str] = None,
                         max_sequence_length: int = 160000,
                         cache_dir: str = "./cache/ncbi") -> NCBIGeneProcessor:
    """创建NCBI处理器的便捷函数"""
    config = NCBIConfig(
        email=email,
        api_key=api_key,
        max_sequence_length=max_sequence_length,
        cache_dir=cache_dir
    )
    return NCBIGeneProcessor(config)

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建处理器（需要设置你的邮箱）
    processor = create_ncbi_processor(email="test@example.com")

    # 测试小数据集
    test_gene_ids = ["1", "2", "3", "4", "5"]  # 几个测试基因ID

    gene_triplets = processor.process_gene_dataset(test_gene_ids)

    print(f"处理结果: {len(gene_triplets)} 个基因三胞胎")
    if gene_triplets:
        example = gene_triplets[0]
        print(f"示例基因: {example.gene_metadata['gene_symbol']}")
        print(f"DNA序列长度: {len(example.dna_sequence)}")
        print(f"目标描述: {example.target_description}")
        print(f"提示: {example.prompt}")

    # 提示：完整下载需要大量时间和API配额
    # full_dataset = processor.download_comprehensive_dataset(limit=1000)