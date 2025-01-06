import React, { useState } from 'react';
import styles from './BookModal.module.css'; 

const BookModal = ({ book, onClose, onSave }) => {
  const [formData, setFormData] = useState({
    bookId: book?.bookId || '',
    title: book?.title || '',
    author: book?.author || '',
    publisher: book?.publisher || '',
    publicationYear: book?.publicationYear || '',
    isbn: book?.isbn || '',
    genre: book?.genre || '',
    descriptions: book?.descriptions || '',
    coverImageUrl: book?.coverImageUrl || '',
    quantity: book?.quantity || '',
    availableQuantity: book?.availableQuantity || '',
    previewLink: book?.previewLink || '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <div className={styles.modalOverlay}>
      <div className={styles.modalContent}>
        <h3>{book ? 'Edit Book' : 'Add New Book'}</h3>
            <form onSubmit={handleSubmit}>
            <div className={styles.inputGroup}>
                <label htmlFor="title">Title:</label>
                <input
                type="text"
                id="title"
                name="title"
                value={formData.title}
                onChange={handleChange}
                required
                />
            </div>

            <div className={styles.inputGroup}>
                <label htmlFor="author">Author:</label>
                <input
                type="text"
                id="author"
                name="author"
                value={formData.author}
                onChange={handleChange}
                required
                />
            </div>

            <div className={styles.inputGroup}>
                <label htmlFor="publisher">Publisher:</label>
                <input
                type="text"
                id="publisher"
                name="publisher"
                value={formData.publisher}
                onChange={handleChange}
                required
                />
            </div>

            <div className={styles.inputGroup}>
                <label htmlFor="isbn">ISBN:</label>
                <input
                type="text"
                id="isbn"
                name="isbn"
                value={formData.isbn}
                onChange={handleChange}
                required
                />
            </div>

            <div className={styles.inputGroup}>
                <label htmlFor="genre">Genre:</label>
                <input
                type="text"
                id="genre"
                name="genre"
                value={formData.genre}
                onChange={handleChange}
                required
                />
            </div>

            <div className={styles.inputGroup}>
                <label htmlFor="descriptions">Descriptions:</label>
                <textarea
                id="descriptions"
                name="descriptions"
                value={formData.descriptions}
                onChange={handleChange}
                required
                />
            </div>

            <div className={styles.inputGroup}>
                <label htmlFor="coverImageUrl">Cover Image URL:</label>
                <input
                type="text"
                id="coverImageUrl"
                name="coverImageUrl"
                value={formData.coverImageUrl}
                onChange={handleChange}
                />
            </div>

            <div className={styles.inputGroup}>
                <label htmlFor="quantity">Quantity:</label>
                <input
                type="number"
                id="quantity"
                name="quantity"
                value={formData.quantity}
                onChange={handleChange}
                required
                />
            </div>

            <div className={styles.inputGroup}>
                <label htmlFor="availableQuantity">Available Quantity:</label>
                <input
                type="number"
                id="availableQuantity"
                name="availableQuantity"
                value={formData.availableQuantity}
                onChange={handleChange}
                required
                />
            </div>

            {/* <div className={styles.inputGroup}>
                <label htmlFor="createdAt">Created At:</label>
                <input
                type="datetime-local"
                id="createdAt"
                name="createdAt"
                value={formData.createdAt}
                onChange={handleChange}
                required
                />
            </div> */}

            <div className={styles.modalActions}>
                <button type="submit">Save</button>
                <button type="button" onClick={onClose}>Cancel</button>
            </div>
            </form>


      </div>
    </div>
  );
};

export default BookModal;
