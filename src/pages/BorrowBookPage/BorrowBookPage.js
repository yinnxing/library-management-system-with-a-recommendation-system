import { useParams, useNavigate } from 'react-router-dom';
import books from '../../assets/books';
import styles from './BorrowBookPage.module.css'; // Tùy chọn: CSS module để styling
import React, { useState } from 'react';
import { useUser } from '../../contexts/UserContext'; // Import useUser hook
import UserApi from '../../api/UserApi';

const BorrowBookPage = () => {
    const { bookId } = useParams();
    const { user } = useUser();
    const book = books.find((b) => String(b.bookId) === bookId);

    const [borrowStatus, setBorrowStatus] = useState({});

    if (!book) {
        return <p>Book not found.</p>;
    }

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

            {borrowStatus && (
                <div className={styles.borrowSuccess}>
                    <p>{borrowStatus.message}</p>
                    <p><strong>Expiry Date for Offline Borrow:</strong> {borrowStatus.expiryDate}</p>
                    <p><strong>Library Address:</strong> {borrowStatus.libraryAddress}</p>
                    <p><strong>Contact Information:</strong> {borrowStatus.contactInfo}</p>
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
