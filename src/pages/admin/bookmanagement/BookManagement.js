import React, { useState, useEffect } from 'react';
import styles from './BookManagement.module.css'; 
import BookModal from './BookModal'; 
import AdminApi from '../../../api/AdminApi';

const BookManagement = () => {
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [currentBook, setCurrentBook] = useState(null);

  useEffect(() => {
    fetchBooks();
  }, []);

  const fetchBooks = () => {
    setLoading(true);
    AdminApi.getBooks()
      .then((response) => {
        setBooks(response.data.result.content);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching books:', error);
        setLoading(false);
      });
  };

  const handleDelete = (bookId) => {
    if (window.confirm('Are you sure you want to delete this book?')) {
      AdminApi.deleteBook(bookId)
        .then(() => {
          setBooks((prevBooks) => prevBooks.filter((book) => book.bookId !== bookId));
          alert('Book deleted successfully!');
        })
        .catch((error) => console.error('Error deleting book:', error));
    }
  };

  const handleEdit = (book) => {
    setCurrentBook(book); // Đặt sách hiện tại vào modal để chỉnh sửa
    setShowModal(true);
  };

  const handleAdd = () => {
    setCurrentBook(null); // Mở modal ở chế độ thêm mới
    setShowModal(true);
  };

  const handleModalClose = () => {
    setShowModal(false);
    setCurrentBook(null);
  };

const handleSave = (updatedBook) => {
  if (updatedBook.bookId) {
    AdminApi.editBook(updatedBook.bookId, updatedBook)
      .then((response) => {
        setBooks((prevBooks) =>
          prevBooks.map((book) =>
            book.bookId === response.data.result.bookId ? response.data.result : book
          )
        );
        handleModalClose();
        alert('Book updated successfully!');
      })
      .catch((error) => {
        console.error('Error updating book:', error);
        alert('Failed to update book. Please try again.');
      });
  } else {
    AdminApi.createBook(updatedBook)
      .then((response) => {
        setBooks((prevBooks) => [response.data.result, ...prevBooks]);
        handleModalClose();
        alert('Book added successfully!');
      })
      .catch((error) => {
        console.error('Error adding book:', error);
        alert('Failed to add book. Please try again.');
      });
  }
};


  return (
    <div className={styles.bookManagementContainer}>
      <h2>Book Management</h2>
      <button onClick={handleAdd} className={styles.addButton}>Add New Book</button>

      {loading ? (
        <p>Loading books...</p>
      ) : (
        books.length > 0 ? (
          <table className={styles.bookTable}>
            <thead>
              <tr>
                <th>Title</th>
                <th>Author</th>
                <th>Publisher</th>
                <th>ISBN</th>
                <th>Quantity</th>
                <th>Available Quantity</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {books.map((book) => (
                <tr key={book.bookId}>
                  <td>{book.title}</td>
                  <td>{book.author}</td>
                  <td>{book.publisher}</td>
                  <td>{book.isbn}</td>
                  <td>{book.quantity}</td>
                  <td>{book.availableQuantity}</td>
                  <td>
                    <button className={styles.editButton} onClick={() => handleEdit(book)}>Edit</button>
                    <button className={styles.deleteButton} onClick={() => handleDelete(book.bookId)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>No books available.</p>
        )
      )}

      {showModal && (
        <BookModal
          book={currentBook}
          onClose={handleModalClose}
          onSave={handleSave}
        />
      )}
    </div>
  );
};

export default BookManagement;
