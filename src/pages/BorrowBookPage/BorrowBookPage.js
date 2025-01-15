import { useParams, useNavigate } from 'react-router-dom';
import books from '../../assets/books';
import styles from './BorrowBookPage.module.css'; 
import React, { useState, useEffect } from 'react';
import { useUser } from '../../contexts/UserContext';
import UserApi from '../../api/UserApi';


const BorrowBookPage = () => {
    const { bookId } = useParams();
    const { user } = useUser();
    const [book, setBook] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [borrowStatus, setBorrowStatus] = useState({});

    useEffect(() => {
        const fetchBook = async () => {
            try {
                setLoading(true);
                const response = await UserApi.getBook(bookId);
                setBook(response.data.result); 
                setLoading(false);
            } catch (err) {
                console.error("Error fetching book:", err);
                setError("Failed to load book information. Please try again later.");
                setLoading(false);
            }
        };

        fetchBook();
    }, [bookId]);

   const handleBorrowBook = async () => {
    try {
        console.log(user.userId);
        const response = await UserApi.borrow(user.userId, parseInt(bookId));
        console.log(response);

        const { expiryDate } = response.data.result;
        console.log(expiryDate);

        setBorrowStatus({
            success: true,
            message: "Your online borrow request is successful!",
            expiryDate: expiryDate,
            libraryAddress: "123 Library Street, City, Country", 
            contactInfo: "+123456789 / library@example.com",
        });
    } catch (error) {
        console.error("Error borrowing book:", error);
        setBorrowStatus({
            success: false,
            message: "Failed to borrow the book. Please try again later.",
        });
    }
};


    if (loading) {
        return <p>Loading book information...</p>;
    }

    if (error) {
        return <p>{error}</p>;
    }

    if (!book) {
        return <p>Book not found.</p>;
    }

    return (
        <div className={styles.borrowPageContainer}>
            <h1 className={styles.title}>{book.title}</h1>

            <div className={styles.bookInfo}>
                <img
                    src={book.coverImageUrl}
                    alt={book.title}
                    className={styles.bookImage}
                />

                <div className={styles.bookDetails}>
                    <p><strong>Author:</strong> {book.author}</p>
                    <p><strong>Available Copies:</strong> {book.availableQuantity}</p>
                </div>
            </div>

            <button
                onClick={handleBorrowBook}
                disabled={book.availableQuantity === 0}
                className={styles.borrowButton}
            >
                Borrow Book
            </button>

            {borrowStatus.message && (
                <div className={styles.borrowSuccess}>
                    <p>{borrowStatus.message}</p>
                    {borrowStatus.expiryDate && (
                        <>
                            <p><strong>Expiry Date for Offline Borrow:</strong> {borrowStatus.expiryDate}</p>
                            <p><strong>Library Address:</strong> {borrowStatus.libraryAddress}</p>
                            <p><strong>Contact Information:</strong> {borrowStatus.contactInfo}</p>
                        </>
                    )}
                    <p className={styles.borrowNote}>
                        Please visit the library within 7 days to complete the borrowing process.
                        If you fail to do so, your request will be canceled automatically.
                    </p>
                </div>
            )}
        </div>
    );
};

export default BorrowBookPage;
