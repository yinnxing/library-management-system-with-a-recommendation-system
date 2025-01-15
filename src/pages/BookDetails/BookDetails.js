import React from "react";
import {useState, useEffect } from 'react';
import styles from "./BookDetails.module.css";
import { useParams } from "react-router-dom";
import books from '../../assets/books'
import UserApi from '../../api/UserApi';


const BookDetails = () => {
    const { bookId } = useParams(); 
    const [book, setBook] = useState(null); 
    const [loading, setLoading] = useState(true); 
    const [error, setError] = useState(null); 

    useEffect(() => {
        const fetchBookDetails = async () => {
            try {
                setLoading(true); 
                const response = await UserApi.getBook(bookId); 
                const data = response.data.result;
                if (!data) {
                    throw new Error('Failed to fetch book details');
                }
                setBook(data); 
            } catch (error) {
                setError(error.message); 
            } finally {
                setLoading(false); 
            }
        };

        fetchBookDetails();
    }, [bookId]); 

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    const {
        title,
        author,
        publisher,
        publicationYear,
        isbn,
        genre,
        descriptions,
        coverImageUrl,
        quantity,
        availableQuantity,
        previewLink,
    } = book || {}; 

    return (
        <div className={styles.container}>
            {/* Hiển thị ảnh bìa sách */}
            <img
                src={coverImageUrl}
                alt={`${title} cover`}
                className={styles.coverImage}
            />
            <div className={styles.infoContainer}>
                <h2>{title}</h2>
                <p><strong>Author:</strong> {author}</p>
                <p><strong>Publisher:</strong> {publisher}</p>
                <p><strong>Year of Publication:</strong> {publicationYear}</p>
                <p><strong>ISBN:</strong> {isbn}</p>
                <p><strong>Genre:</strong> {genre || "General"}</p>
                <p className={styles.description}><strong>Description:</strong> {descriptions || "No description available."}</p>
                <p><strong>Total Quantity:</strong> {quantity}</p>
                <p><strong>Available Quantity:</strong> {availableQuantity}</p>
                <iframe src={previewLink}
                width="700" height="900"></iframe>

            </div>
        </div>
    );
};

export default BookDetails;
