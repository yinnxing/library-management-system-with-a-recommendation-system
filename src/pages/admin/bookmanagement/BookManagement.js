import React, { useState, useEffect } from 'react';
import styles from './BookManagement.module.css'; 
import BookModal from './BookModal'; 
import all_books from '../../../assets/books'
const BookManagement = () => {
  const [books, setBooks] = useState(all_books);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [currentBook, setCurrentBook] = useState(null);

  useEffect(() => {
    fetch('/api/admin/books')
      .then((response) => response.json())
      .then((data) => {
        setBooks(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching books:', error);
        setLoading(false);
      });
  }, []);

  const handleDelete = (bookId) => {
    const confirmDelete = window.confirm('Are you sure you want to delete this book?');
    if (confirmDelete) {
      fetch(`/api/admin/books/${bookId}`, {
        method: 'DELETE',
      })
        .then(() => {
          setBooks(books.filter(book => book.bookId !== bookId));
        })
        .catch((error) => console.error('Error deleting book:', error));
    }
  };

  const handleEdit = (book) => {
    setCurrentBook(book);
    setShowModal(true);
  };

  const handleAdd = () => {
    setCurrentBook(null);
    setShowModal(true);
  };

  const handleModalClose = () => {
    setShowModal(false);
    setCurrentBook(null);
  };

  const handleSave = (updatedBook) => {
    if (updatedBook.bookId) {
      fetch(`/api/admin/books/${updatedBook.bookId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedBook),
      })
        .then((response) => response.json())
        .then((data) => {
          setBooks(books.map(book => (book.bookId === data.bookId ? data : book)));
          handleModalClose();
        })
        .catch((error) => console.error('Error updating book:', error));
    } else {
      fetch('/api/admin/books', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedBook),
      })
        .then((response) => response.json())
        .then((data) => {
          setBooks([...books, data]);
          handleModalClose();
        })
        .catch((error) => console.error('Error adding book:', error));
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