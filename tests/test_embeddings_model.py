"""Tests de src.embeddings.model : normalisation de la casse avant embedding."""

from src.embeddings.model import encoder, normaliser_pour_embedding


def test_normaliser_pour_embedding_met_en_minuscules():
    assert normaliser_pour_embedding("ISI") == "isi"
    assert normaliser_pour_embedding("C'est quoi ISI ?") == "c'est quoi isi ?"


def test_isi_majuscule_et_minuscule_produisent_le_meme_vecteur():
    vecteur_majuscule = encoder([normaliser_pour_embedding("C'est quoi ISI ?")])[0]
    vecteur_minuscule = encoder([normaliser_pour_embedding("c'est quoi isi ?")])[0]

    assert vecteur_majuscule == vecteur_minuscule
